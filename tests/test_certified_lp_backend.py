"""CPU correctness of optional certified LP; no mock GPU numeric success."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from lp_contract import CanonicalLP
from lp_backends import RawResult,solve_lp
import lp_backends as adapters
from lp_certify import certify
import certified_pgfp as pg
from probes import gpu_lp_solver_probe as p7
from probes import pgfp_dual_stability as g
from probes import pgfp_gpu_scoring as p6


def contract(m):return CanonicalLP(m['A'],m['c'],m['l'],m['u'],m['lb'],m['ub'],m['offset'])


@pytest.mark.parametrize('m',[
 p7.smoke_models()['hand_mixed'],
 p7.model([1,-2],np.empty((0,2)),[],[],[-3,2],[1,4],2.),
 p7.model([1,0],[[1,1],[1,0]],[2,0],[3,4],[-np.inf,-np.inf],[np.inf,np.inf]),
 p7.model([1],[[1]],[1],[2],[0],[np.inf])])
def test_highs_ranged_free_bound_offset_signs(m):
    lp=contract(m);result=solve_lp(lp)
    assert result.status=='OPTIMAL' and result.certified
    assert result.attempts[0].certificate['native_objective_fields_match']
    assert np.max(np.abs(lp.c-lp.A.T@result.raw.pi-result.raw.rc),initial=0.)<=1e-7


def test_canonical_owns_data_hash_readonly_and_bad_models():
    m=p7.smoke_models()['hand_mixed'];lp=contract(m);source=lp.source_hash;m['c'][0]=999;m['A'].data[0]=999
    assert lp.source_hash==source and lp.c[0]==1.
    with pytest.raises(ValueError):lp.c[0]=5.
    for key,value in [('c',np.array([np.nan,1.])),('l',np.array([np.inf,-np.inf,-np.inf])),('ub',np.array([-np.inf,np.inf]))]:
        bad=p7.smoke_models()['hand_mixed'];bad[key]=value
        with pytest.raises(ValueError):contract(bad)
    A=csr_matrix([[1.]]);A.indices[0]=2
    with pytest.raises(ValueError):CanonicalLP(A,[1],[-np.inf],[1],[0],[np.inf])
    bad=p7.smoke_models()['hand_mixed'];bad['A']=np.eye(2)
    with pytest.raises(ValueError):contract(bad)
    lp.c.flags.writeable=True;lp.c[0]=8.
    with pytest.raises(ValueError):solve_lp(lp)


def test_false_optimal_missing_wrong_native_fields_and_bound_dual():
    lp=contract(p7.smoke_models()['hand_mixed']);good=RawResult('fixture','analytic','OPTIMAL',x=np.array([1.,2.]),pi=np.array([1.,-1.,0.]),rc=np.zeros(2),objective=5.,dual_objective=5.,gap=0.)
    assert certify(lp,good)['certified']
    for raw in (replace(good,x=None),replace(good,x=np.array([1.,2.+2e-7])),replace(good,pi=-good.pi),replace(good,rc=np.ones(2)),replace(good,dual_objective=6.),replace(good,gap=1.),replace(good,status='TIME_LIMIT')):assert not certify(lp,raw)['certified']


def test_terminal_ray_certification_and_missing_rays_never_retried(monkeypatch):
    lp=contract(p7.smoke_models()['infeasible']);ray=RawResult('fixture','analytic','INFEASIBLE',pi=np.array([1.,-1.]),rc=np.zeros(1))
    assert certify(lp,ray)['certified']
    monkeypatch.setattr(adapters,'_highs',lambda *_:ray);out=solve_lp(lp,fallback='highs');assert out.status=='INFEASIBLE' and len(out.attempts)==1
    monkeypatch.setattr(adapters,'_highs',lambda *_:replace(ray,pi=None,rc=None));out=solve_lp(lp,fallback='highs');assert out.status=='NOT_CERTIFIED_INFEASIBLE' and len(out.attempts)==1
    ub=contract(p7.smoke_models()['unbounded']);assert certify(ub,RawResult('fixture','analytic','UNBOUNDED',x=np.array([1.])) )['certified']
    harder=CanonicalLP(csr_matrix([[1.]]),[-1.],[2.],[np.inf],[2.],[np.inf])
    assert not certify(harder,RawResult('fixture','analytic','UNBOUNDED',x=np.array([1.])))['certified']
    assert certify(harder,RawResult('fixture','analytic','UNBOUNDED',x=np.array([1.]),witness=np.array([2.])))['certified']


def test_explicit_fallback_unavailable_error_timeout_and_cpu_failure(monkeypatch):
    lp=contract(p7.smoke_models()['hand_mixed'])
    for status in ('BACKEND_UNAVAILABLE','ERROR','TIME_LIMIT','NUMERIC_UNCERTIFIED'):
        monkeypatch.setattr(adapters,'_hpdlp',lambda *_,status=status:RawResult('hpdlp',None,status,error='deliberate unavailable/error path'))
        assert solve_lp(lp,backend='hpdlp').status==status
        result=solve_lp(lp,backend='hpdlp',fallback='highs');assert result.status=='OPTIMAL' and len(result.attempts)==2 and result.raw.backend=='highs'
        assert result.elapsed_seconds>=sum(x.elapsed_seconds for x in result.attempts)
    monkeypatch.setattr(adapters,'_highs',lambda *_:RawResult('highs',None,'ERROR',error='CPU failure'))
    result=solve_lp(lp,backend='hpdlp',fallback='highs');assert result.status=='ERROR' and not result.certified and len(result.attempts)==2
    with pytest.raises(ValueError):solve_lp(lp,backend='auto')
    with pytest.raises(ValueError):solve_lp(lp,time_limit=1+2j)


def test_numeric_uncertified_cpu_attempt_does_not_recurse(monkeypatch):
    lp=contract(p7.smoke_models()['hand_mixed']);monkeypatch.setattr(adapters,'_highs',lambda *_:RawResult('fixture',None,'OPTIMAL',objective=0.))
    result=solve_lp(lp,backend='highs',fallback='highs');assert result.status=='NUMERIC_UNCERTIFIED' and len(result.attempts)==1


def test_cpu_only_import_and_use_does_not_import_optional_gpu():
    code="""import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 missing=False
 def find_spec(self,fullname,*args):
  if fullname=='hpdlp':
   if self.missing:raise ModuleNotFoundError('explicit CPU-only missing optional backend')
   raise AssertionError('optional GPU import on CPU')
sys.meta_path.insert(0,Block())
from lp_contract import CanonicalLP
from lp_backends import solve_lp
from scipy.sparse import csr_matrix
p=CanonicalLP(csr_matrix([[1.]]),[-1.],[-float('inf')],[1.],[0.],[float('inf')])
assert solve_lp(p).certified
assert 'hpdlp' not in sys.modules
Block.missing=True
r=solve_lp(p,backend='hpdlp',fallback='highs')
assert r.certified and len(r.attempts)==2 and r.attempts[0].status=='BACKEND_UNAVAILABLE'
"""
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


def test_original_p5_1k_sparse_pin_and_strict_certificate():
    a=p7.p5.fixture(1000,64);model,labels=pg.canonical_shape(a);old=p7.p5.assemble(a)
    assert labels==list(range(8)) and model.A.nnz==32000 and model.A.shape==(1024,8000)
    assert p7.p5.digest(model.A.data,model.A.indices,model.A.indptr,model.row_hi,model.c)==p7.PINS[1000][1]
    result=solve_lp(model);assert result.certified and p7.p5.certify(old,result.raw.x,result.raw.pi,result.raw.rc,True,result.raw.objective)['certified']


@pytest.mark.parametrize('a',[*g.scenarios().values(),g.p4b.fixed_cases()[0][1],p6.edge_case(),g.oracle.packing_case()])
def test_default_and_optin_shadow_full_ledger_physics(a):
    out=pg.shadow(a);assert out['accepted'] and out['lp']['result'].certified
    for path in ('proposed','baseline'):
        q=out[path];rebuilt=g.audit_steps(a,q['ledger_steps'],q['task_order'],q['shape_order']);assert rebuilt['placement']==q['placement'] and rebuilt['future_score']==q['future_score']
    actual=g.api.solve(a,method='pgfp',seed=11,time_limit=.5)
    assert tuple(out['baseline']['placement'])==actual.placement and out['baseline']['value']==actual.value
    strict=pg.shadow(a,strict_compatibility=True);assert strict['accepted']==all(strict['comparison'].values())


def test_strict_shadow_rejects_alternate_certified_prices(monkeypatch):
    a=g.scenarios()['A_analytic_split'];base=pg.shadow(a);model,_=pg.canonical_shape(a)
    raw=RawResult('analytic_fixture','independent','OPTIMAL',x=np.ones(2),pi=np.array([0.,0.,-1.,0.,0.]),rc=np.zeros(2),objective=-3.,dual_objective=-3.,gap=0.)
    assert certify(model,raw)['certified']
    alternative=dict(base['lp'],prices=-raw.pi[2:].reshape(1,3))
    # This fixture has a certified dual face; force only the test wrapper result,
    # not a mocked GPU numerical success or a production solver modification.
    monkeypatch.setattr(pg,'shape_lp',lambda *args,**kw:alternative)
    result=pg.shadow(a,strict_compatibility=True)
    assert result['status']=='STRICT_COMPATIBILITY_REJECTED' and not result['accepted']


def test_single_physical_node_default_order_and_rejection():
    a=g.oracle.Instance(np.full((1,3),2.),np.ones((3,3)),np.array([2.,1.,1.]),np.ones((1,3),bool),np.array([0]),np.ones((1,3)),np.ones(1),np.ones((1,1),bool))
    out=pg.shadow(a);assert out['proposed']['placement']==[0,0,-1] and out['proposed']['future_score']==0.
    assert out['baseline']['placement']==list(g.api.solve(a,method='pgfp',seed=11,time_limit=.5).placement)
    q=out['proposed'];assert g.audit_steps(a,q['ledger_steps'],q['task_order'],q['shape_order'])['value']==3.


def test_large_shadow_never_enters_dense_legacy_baseline(monkeypatch):
    original=g.scenarios()['C_p4f_128'];a=replace(original,demand=np.tile(original.demand,(4,1)),value=np.tile(original.value,4),eligible=np.tile(original.eligible,(1,4)))
    def forbidden(*args,**kwargs):raise AssertionError('large shadow must fail before any LP/dense baseline')
    monkeypatch.setattr(pg,'shape_lp',forbidden);monkeypatch.setattr(pg.oracle,'shape_lp',forbidden)
    with pytest.raises(ValueError,match='1M dense entries'):pg.shadow(a)
