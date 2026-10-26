"""Deterministic source-deposit DRAFT from a frozen Git commit; never submits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def paginate(files,lines_per_page=50):
    """Original logical lines only, no padding. UTF-8 render uses LF."""
    lines=[]
    for name,raw in files:
        lines.extend((name,i,line) for i,line in enumerate(raw.decode('utf-8').splitlines(),1))
    return [lines[i:i+lines_per_page] for i in range(0,len(lines),lines_per_page)]


def selected_pages(count):return list(range(count)) if count<=60 else list(range(30))+list(range(count-30,count))


def generate(root,commit,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.is_relative_to(root):raise ValueError('output must be outside Git checkout')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('full immutable commit SHA required')
    root=Path(subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel']).decode().strip()).resolve()
    if output.is_relative_to(root):raise ValueError('output must be outside actual Git checkout')
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args])
    if git('cat-file','-t',commit).strip()!=b'commit':raise ValueError('not a commit')
    config=git('show',commit+':pyproject.toml').decode();version=re.search(r'^version\s*=\s*"([^"]+)"',config,re.M).group(1)
    if version!='1.0.0':raise ValueError('expected V1.0 package1.0.0 candidate')
    modules=ast.literal_eval(re.search(r'py-modules\s*=\s*(\[[\s\S]*?\])',config).group(1));packages=ast.literal_eval(re.search(r'^packages\s*=\s*(\[[^\n]*\])',config,re.M).group(1))
    if not all(isinstance(n,str) and re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*',n) for n in modules+packages):raise ValueError('unsafe source scope')
    tree=git('ls-tree','-r',commit).decode().splitlines();entries={line.split('\t',1)[1]:line.split('\t',1)[0].split() for line in tree};names=[m.replace('.','/')+'.py' for m in modules]
    for package in packages:names.extend(sorted(n for n in entries if n.startswith(package.replace('.','/')+'/') and n.endswith('.py')))
    if len(set(names))!=len(names):raise ValueError('duplicate source scope')
    files=[];manifest=[]
    for name in names:
        mode,kind,blob=entries[name]
        if mode not in ('100644','100755') or kind!='blob':raise ValueError('regular source blobs only')
        raw=git('show',commit+':'+name);files.append((name,raw));manifest.append(dict(path=name,blob=blob,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    pages=paginate(files);selection=selected_pages(len(pages));output.mkdir(parents=True,exist_ok=False);page_info=[]
    for index in selection:
        page=pages[index];raw=('\n'.join(x[2] for x in page)+'\n').encode();filename=f'source-{index+1:03}.txt';(output/filename).write_bytes(raw)
        page_info.append(dict(page=index+1,file=filename,lines=len(page),sha256=hashlib.sha256(raw).hexdigest(),source_lines=[dict(path=n,line=i) for n,i,_ in page]))
    result=dict(status='DRAFT_NOT_FILING_APPROVAL',software='AccelSched 异构算力资源优化调度软件 V1.0',package_version=version,commit=commit,tree=git('rev-parse',commit+'^{tree}').decode().strip(),source_scope='declared runtime py-modules and examples Python package; excludes tests/probes/tools/dependencies/data',source_files=manifest,total_pages=len(pages),selected_page_numbers=[i+1 for i in selection],pages=page_info,rendering='50 original logical lines/page, no added padding; LF rendering; short final page and print layout require manual review; header/name/version/page number come from manifest, not fake source lines',ownership='ru zhang per user declaration; applicant identity/contribution clearance and official acceptance not verified')
    (output/'manifest.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.');p.add_argument('--commit',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=generate(a.repo,a.commit,a.output);print(json.dumps(dict(status=r['status'],commit=r['commit'],total_pages=r['total_pages'],selected_pages=len(r['pages']))))
