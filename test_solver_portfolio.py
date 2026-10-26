"""CPU portfolio contracts; mocked GPU paths are failures only."""
import importlib
import sys
import time
import numpy as np
import pytest
from scipy.sparse import csr_matrix
import solver_portfolio as p
import lp_backends as b
from lp_contract import CanonicalLP
from certified_pgfp import shadow
from probes import pgfp_dual_stability as g

def model():return CanonicalLP(csr_matrix([[1.]]),[-1.],[-np.inf],[2.],[0.],[np.inf],3.)

def test_cpu_policy_no_gpu_discovery_and_accounting(monkeypatch):
    monkeypatch.setattr(p.importlib.util,'find_spec',lambda *_:pytest.fail('CPU optional discovery'))
    a=p.solve_portfolio(model(),device={'cuda':False});c=p.solve_portfolio(model(),device={'cuda':False})
    assert a.certified and a.raw.backend=='highs' and a.selection==c.selection and a.selection['order']==['highs']
    q=a.profile;assert abs(q['selection_wall']+q['attempt_wall']+q['accounting_wall']-q['total_wall'])<1e-9
    assert q['fallback_wall']==0 and a.source_hash==model().source_hash
    assert 'cupdlpx' not in sys.modules and 'hpdlp' not in sys.modules

def test_gpu_hints_order_override_and_no_duplicates(monkeypatch):
    monkeypatch.setattr(p.importlib.util,'find_spec',lambda *_:object())
    for name in ('hpdlp','cupdlpx'):monkeypatch.setattr(b,'_'+name,lambda *_,n=name:b.RawResult(n,None,'BACKEND_UNAVAILABLE'))
    q=p.solve_portfolio(model(),device={'cuda':True},warm=True,candidates=['cupdlpx','cupdlpx','hpdlp'])
    assert q.selection['order']==['cupdlpx','hpdlp','highs'] and [a.raw.backend for a in q.attempts]==q.selection['order'] and q.certified
    assert p.solve_portfolio(model(),device={'cuda':True}).raw.backend=='highs'
    explicit=p.solve_portfolio(model(),policy='hpdlp',device={'cuda':False},candidates=['cupdlpx'])
    assert explicit.attempts[0].raw.backend=='hpdlp' and explicit.certified
    none=p.solve_portfolio(model(),policy='cupdlpx',fallback=False,device={'cuda':False})
    assert len(none.attempts)==1 and not none.certified

@pytest.mark.parametrize('status',['ERROR','TIME_LIMIT','OPTIMAL','INFEASIBLE','UNBOUNDED'])
def test_failures_and_uncertified_terminal(monkeypatch,status):
    monkeypatch.setattr(b,'_cupdlpx',lambda *_:b.RawResult('cupdlpx',None,status,error='injected no vectors'))
    out=p.solve_portfolio(model(),policy='cupdlpx',device={'cuda':False})
    terminal=status in ('INFEASIBLE','UNBOUNDED')
    assert len(out.attempts)==(1 if terminal else 2)
    assert out.certified== (not terminal)
    if terminal:assert out.status=='NOT_CERTIFIED_'+status

def test_certified_terminal_rays_are_terminal(monkeypatch):
    m=CanonicalLP(csr_matrix([[1.],[1.]]),[0.],[2.,-np.inf],[np.inf,1.],[0.],[np.inf])
    monkeypatch.setattr(b,'_cupdlpx',lambda *_:b.RawResult('cupdlpx','analytic-ray','INFEASIBLE',pi=np.array([1.,-1.]),rc=np.zeros(1)))
    q=p.solve_portfolio(m,policy='cupdlpx',device={'cuda':False});assert q.certified and q.status=='INFEASIBLE' and len(q.attempts)==1
    u=CanonicalLP(csr_matrix((0,1)),[-1.],[],[],[0.],[np.inf])
    monkeypatch.setattr(b,'_cupdlpx',lambda *_:b.RawResult('cupdlpx','analytic-ray','UNBOUNDED',x=np.ones(1)))
    q=p.solve_portfolio(u,policy='cupdlpx',device={'cuda':False});assert q.certified and q.status=='UNBOUNDED' and len(q.attempts)==1

def test_budget_exhaustion_and_native_clamp(monkeypatch):
    def delayed(*_):time.sleep(.01);return b.RawResult('cupdlpx',None,'TIME_LIMIT')
    monkeypatch.setattr(b,'_cupdlpx',delayed)
    q=p.solve_portfolio(model(),policy='cupdlpx',wall_budget=.005,device={'cuda':False})
    assert q.status=='BUDGET_EXHAUSTED' and len(q.attempts)==1 and q.profile['attempts'][0]['native_time_limit']<=.005
    q=p.solve_portfolio(model(),wall_budget=1e-12,device={'cuda':False});assert not q.certified and q.raw is None and q.status=='BUDGET_EXHAUSTED'

def test_invalid_empty_and_shadow_callback(monkeypatch):
    for kwargs in ({'policy':'magic'},{'candidates':[]},{'candidates':'highs'},{'time_limit':0},{'wall_budget':float('nan')},{'device':{'cuda':'yes'}}):
        with pytest.raises(ValueError):p.solve_portfolio(model(),**kwargs)
    q=p.solve_portfolio(model(),candidates=['cupdlpx'],fallback=False,device={'cuda':False});assert q.status=='NO_AVAILABLE_BACKEND' and not q.certified
    a=g.scenarios()['A_analytic_split'];out=shadow(a,solver=lambda m:p.solve_portfolio(m,device={'cuda':False}),strict_compatibility=True)
    assert out['accepted'] and out['status']=='EXACT_BASELINE'
    def wrong(m):
        q=p.solve_portfolio(m,device={'cuda':False});q.source_hash='wrong';return q
    with pytest.raises(ValueError):shadow(a,solver=wrong)
