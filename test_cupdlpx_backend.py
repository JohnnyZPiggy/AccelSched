"""Optional third backend: CPU tests simulate failures, never GPU numeric success."""
from dataclasses import replace
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix
import lp_backends as b
from lp_contract import CanonicalLP
import certified_pgfp as pg
from probes import pgfp_dual_stability as g


def model():return CanonicalLP(csr_matrix([[1.]]),[-1.],[-np.inf],[2.],[0.],[np.inf],3.)


@pytest.mark.parametrize('error',[ImportError('missing'),OSError('CUDA unavailable')])
def test_lazy_unavailable_cpu_fallback(monkeypatch,error):
    original=b.importlib.import_module;seen=[]
    def missing(name):
        seen.append(name)
        if name=='cupdlpx':raise error
        return original(name)
    monkeypatch.setattr(b.importlib,'import_module',missing)
    out=b.solve_lp(model(),backend='cupdlpx',fallback='highs')
    assert out.status=='OPTIMAL' and out.certified and len(out.attempts)==2
    assert out.attempts[0].status=='BACKEND_UNAVAILABLE' and str(error) in out.attempts[0].raw.error
    assert out.raw.backend=='highs' and out.raw.objective==1. and seen==['cupdlpx']
    assert b.solve_lp(model(),backend='cupdlpx').status=='BACKEND_UNAVAILABLE'


@pytest.mark.parametrize('status',['ERROR','TIME_LIMIT','OPTIMAL','INFEASIBLE','UNBOUNDED'])
def test_failed_attempt_preserved_and_terminal_not_retried(monkeypatch,status):
    raw=b.RawResult('cupdlpx','failure-fixture',status,error='simulated failure; no vectors')
    monkeypatch.setattr(b,'_cupdlpx',lambda *_:raw)
    out=b.solve_lp(model(),backend='cupdlpx',fallback='highs')
    assert out.attempts[0].raw is raw and not out.attempts[0].certificate['certified']
    terminal=status in ('INFEASIBLE','UNBOUNDED')
    assert len(out.attempts)==(1 if terminal else 2)
    assert out.status==('NOT_CERTIFIED_'+status if terminal else 'OPTIMAL')
    if status=='OPTIMAL':assert out.attempts[0].status=='NUMERIC_UNCERTIFIED'


def test_exception_no_recursion_and_hpdlp_dispatch_unchanged(monkeypatch):
    def broken(*_):raise RuntimeError('real-style adapter failure')
    monkeypatch.setattr(b,'_cupdlpx',broken)
    high=b._highs;calls=[]
    def counted(*args):calls.append('highs');return high(*args)
    monkeypatch.setattr(b,'_highs',counted)
    out=b.solve_lp(model(),backend='cupdlpx',fallback='highs')
    assert out.certified and len(out.attempts)==2 and calls==['highs']
    assert out.attempts[0].raw.error=='real-style adapter failure'
    monkeypatch.setattr(b,'_hpdlp',lambda *_:b.RawResult('hpdlp',None,'BACKEND_UNAVAILABLE'))
    assert b.solve_lp(model(),backend='hpdlp').raw.backend=='hpdlp'
    monkeypatch.setattr(b,'_highs',broken)
    assert len(b.solve_lp(model(),backend='cupdlpx',fallback='highs').attempts)==2


def test_shadow_reuses_explicit_fallback(monkeypatch):
    monkeypatch.setattr(b,'_cupdlpx',lambda *_:b.RawResult('cupdlpx',None,'TIME_LIMIT'))
    a=g.scenarios()['A_analytic_split'];out=pg.shadow(a,backend='cupdlpx',fallback='highs',strict_compatibility=True)
    assert out['status']=='EXACT_BASELINE' and out['accepted']
    assert len(out['lp']['result'].attempts)==2


def test_cold_cpu_import_never_loads_optional_gpu():
    code="""import builtins,sys
old=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in ('cupdlpx','hpdlp'):raise AssertionError('optional import')
 return old(name,*args,**kwargs)
builtins.__import__=guarded
import lp_backends,certified_pgfp,rolling,accelsched
from scipy.sparse import csr_matrix
from lp_contract import CanonicalLP
import numpy as np
m=CanonicalLP(csr_matrix([[1.]]),[-1.],[-np.inf],[2.],[0.],[np.inf])
assert lp_backends.solve_lp(m).certified
assert 'cupdlpx' not in sys.modules and 'hpdlp' not in sys.modules
"""
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,check=True,timeout=20)
