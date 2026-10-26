import importlib
from pathlib import Path
import subprocess
import pytest
from tools.repository_inventory import inventory


def test_inventory_covers_aliases_relative_edges_paths_and_packaging(tmp_path,monkeypatch):
    (tmp_path/'pkg').mkdir()
    (tmp_path/'pkg/a.py').write_text('import subprocess as sp\nfrom importlib import import_module as load\nfrom pathlib import Path as P\nfrom . import b\nsp.run(["old.py"])\nload("pkg.b")\nP(__file__).resolve()\n',encoding='utf-8')
    (tmp_path/'pkg/b.py').write_text('VALUE=1\n',encoding='utf-8')
    (tmp_path/'pyproject.toml').write_text('[tool.setuptools]\npackages = ["examples"]\npy-modules = ["accelsched"]\n[tool.setuptools.package-data]\nexamples = ["data/*.json"]\n',encoding='utf-8')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'pkg/a.py\npkg/b.py\npyproject.toml\n')
    result=inventory(tmp_path);a=result['files'][0]
    assert result['python_files']==2 and len(a['dynamic_calls'])==4  # P() and P().resolve() are both calls.
    assert any(e['source']=='pkg/a.py' and e['target']=='pkg/b.py' for e in result['edges'])
    assert any(p['value']=='old.py' for p in a['path_literals'])
    assert result['packaging']['package-data']=={'examples':['data/*.json']}


def test_legacy_test_and_historical_function_identity():
    old=importlib.import_module('test_bench_ultra_pgfp');new=importlib.import_module('tests.test_bench_ultra_pgfp')
    assert old.test_cached_exact_index_matches_full_rescan_after_many_updates is new.test_cached_exact_index_matches_full_rescan_after_many_updates
    for name in ['bench_cpu_v3','bench_cpu_v31']:
        old=importlib.import_module(name);new=importlib.import_module('benchmarks.historical.'+name)
        assert old.instance is new.instance and old.run is new.run


def test_repository_inventory_and_installed_scope_stay_explicit():
    root=Path(__file__).resolve().parents[1];result=inventory(root)
    names=subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines()
    assert {f['path'] for f in result['files']}=={n for n in names if n.endswith('.py')}
    assert result['packaging']['packages']==['examples']
    assert result['packaging']['py-modules']==['accelsched','m0_oracle','scipy_backend','workload','rolling','lp_contract','lp_certify','lp_backends','certified_pgfp','solver_portfolio']
    assert not (root/'accelsched').exists()


def test_revision_requires_immutable_commit(tmp_path):
    with pytest.raises(ValueError):inventory(tmp_path,'HEAD')
