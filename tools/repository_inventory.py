"""Deterministic syntax-level inventory of every tracked Python file (no imports).

Dynamic call/path expressions are reported for review, not claimed resolved.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def inventory(root,revision=None):
    root=Path(root).resolve()
    if revision is not None:
        if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('full immutable commit SHA required')
        if subprocess.check_output(['git','-C',str(root),'cat-file','-t',revision]).strip()!=b'commit':raise ValueError('commit required')
        names=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',revision],text=True).splitlines()
        read=lambda n:subprocess.check_output(['git','-C',str(root),'show',revision+':'+n])
    else:
        names=subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines()
        read=lambda n:(root/n).read_bytes()
    python=sorted(n for n in names if n.endswith('.py'))
    modules={n.removesuffix('.py').replace('/','.').removesuffix('.__init__'):n for n in python}
    rows=[];edges=[]
    for name in python:
        raw=read(name);tree=ast.parse(raw,filename=name);imports=[];calls=[];paths=[];fixtures=[];tests=[]
        aliases={}
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):aliases.update({a.asname or a.name.split('.')[0]:a.name for a in n.names})
            elif isinstance(n,ast.ImportFrom):aliases.update({a.asname or a.name:((n.module+'.') if n.module else '')+a.name for a in n.names})
        module=name.removesuffix('.py').replace('/','.');package=module.rsplit('.',1)[0] if '.' in module else ''
        if name.endswith('/__init__.py'):package=module.removesuffix('.__init__')
        for node in ast.walk(tree):
            targets=[]
            if isinstance(node,ast.Import):
                for alias in node.names:imports.append(dict(line=node.lineno,kind='import',module=alias.name,alias=alias.asname));targets.append(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=node.module or ''
                if node.level:
                    parents=package.split('.') if package else []
                    prefix='.'.join(parents[:len(parents)-node.level+1]+([prefix] if prefix else []))
                imports.append(dict(line=node.lineno,kind='from',module=node.module,level=node.level,names=[a.name for a in node.names],resolved_prefix=prefix))
                targets=[prefix]+[prefix+'.'+a.name if prefix else a.name for a in node.names]
            elif isinstance(node,ast.Call):
                function=ast.unparse(node.func)
                head=node.func
                while isinstance(head,(ast.Attribute,ast.Call)):head=head.value if isinstance(head,ast.Attribute) else head.func
                resolved=aliases.get(head.id,head.id) if isinstance(head,ast.Name) else function
                qualified=resolved+function[len(head.id):] if isinstance(head,ast.Name) and function.startswith(head.id) else function
                if any(s in qualified for s in ['import_module','__import__','subprocess','Path','sys.path','run_module','run_path']):calls.append(dict(line=node.lineno,expression=ast.unparse(node),binding=qualified))
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                if any(s in node.value for s in ['.py','.json','.npz','__file__']):paths.append(dict(line=node.lineno,value=node.value))
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                if node.name.startswith('test_'):tests.append(node.name)
                if any('fixture' in ast.unparse(d) for d in node.decorator_list):fixtures.append(node.name)
            for target in targets:
                if target in modules:edges.append(dict(source=name,target=modules[target],line=node.lineno,module=target))
        rows.append(dict(path=name,module=module,sha256=hashlib.sha256(raw).hexdigest(),imports=imports,dynamic_calls=calls,path_literals=paths,pytest_functions=tests,fixtures=fixtures,
            main_guard=any(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test) and '__main__' in ast.unparse(n.test) for n in tree.body)))
    # Same explicit setuptools-list projection as tools/source_deposit.py; no TOML dependency.
    config=read('pyproject.toml').decode('utf-8')
    modules_list=ast.literal_eval(re.search(r'py-modules\s*=\s*(\[[\s\S]*?\])',config).group(1))
    packages=ast.literal_eval(re.search(r'^packages\s*=\s*(\[[^\n]*\])',config,re.M).group(1))
    data={}
    section=re.search(r'\[tool.setuptools.package-data\]\s*([^\[]*(?:\[[^\n]*\][^\n]*\n?)*)',config)
    if section:
        for line in section.group(1).splitlines():
            if '=' in line:
                key,value=line.split('=',1);data[key.strip()]=ast.literal_eval(value.strip())
    documents=[]
    for name in sorted(n for n in names if n.endswith('.md')):
        text=read(name).decode('utf-8')
        documents.append(dict(path=name,links=re.findall(r'\]\(([^)]+)\)',text),commands=[line for line in text.splitlines() if 'python ' in line or 'pytest ' in line]))
    return dict(revision=revision,tracked_files=len(names),python_files=len(rows),files=rows,edges=sorted(edges,key=lambda e:(e['source'],e['line'],e['target'])),
        packaging={'py-modules':modules_list,'packages':packages,'package-data':data},documents=documents,
        pytest=read('pytest.ini').decode('utf-8') if 'pytest.ini' in names else None,
        limitations='AST syntax graph only; SHA256 is Git-blob bytes for a revision, raw working bytes otherwise; dynamic/path expressions require runtime verification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.');p.add_argument('--revision');p.add_argument('--output',required=True);a=p.parse_args()
    result=inventory(a.repo,a.revision);Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(tracked_files=result['tracked_files'],python_files=result['python_files'],edges=len(result['edges']))))
