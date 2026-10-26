from copy import deepcopy
import numpy as np
import pytest
import rolling
from accelsched import METHODS,SolveResult
import bench_v4_dynamic as b


def fixture():
    jobs=[b.Job('A',2,2,(1,4,16),2),b.Job('B',3,1,(1,4,16),7),b.Job('C',4,3,(1,4,16),5)]
    data=b.pack(np.array([[1,4,16]],float),jobs,np.ones((1,3),bool),['A'],['A']*3,
                b.OMEGA,5,1800,case='hand',seed=11,target=None)
    return data,rolling.simulate(*b.unpack(data),'first_fit')


def test_nonuniform_value_clipping_completions_and_units():
    data,r=fixture();m=b.audit(data,r,'first_fit')
    assert m['observed_accepted_value']==7 and m['accepted_count']==2
    assert m['completed_by_T']==1 and m['completed_full_tail']==2
    assert m['full_tail_gpu_hours']==2.5 and m['window_gpu_hours']==1.5
    assert m['window_allocation_ratio']==[.6,.6,.6] and m['allocated_gpu_work_rate']==.6
    assert m['quality_complete'] and m['horizon_optimality_claim'] is False
    assert m['attempted_batches']==3


@pytest.mark.parametrize('mutate',[
    lambda r:r['events'][2]['completed'].clear(),
    lambda r:r['events'][2]['free_before_arrivals'][0].__setitem__(0,0),
    lambda r:r['events'][0]['running_load'][0].__setitem__(0,2),
    lambda r:r['decisions'][2].__setitem__('end',5),
    lambda r:r.__setitem__('accepted_value',2),
    lambda r:r.__setitem__('completed_count',1),
    lambda r:r['batches'][0].__setitem__('optimal',True),
    lambda r:r['events'][0]['free'][0].__setitem__(1,float('nan')),
    lambda r:r.__setitem__('rejected_count',0)])
def test_corruption_rejected(mutate):
    data,r=fixture();mutate(r)
    with pytest.raises(AssertionError):b.audit(data,r,'first_fit')


def test_eligibility_family_and_overlap_rejected():
    data,r=fixture();data['eligible']['values'][0][0]=False
    with pytest.raises(AssertionError):b.audit(data,r,'first_fit')
    data,r=fixture();data['requests'][0]='B'
    with pytest.raises(AssertionError):b.audit(data,r,'first_fit')
    data,r=fixture();r['decisions'][2]['start']=3
    with pytest.raises(AssertionError):b.audit(data,r,'first_fit')


def test_error_not_zero_quality_or_rejection_and_all_latency_samples(monkeypatch):
    data,_=fixture()
    monkeypatch.setattr(rolling,'solve',lambda *args,**kwargs:SolveResult('highs_mip','uncertified',None,None,None,0.,False,'injected'))
    r=rolling.simulate(*b.unpack(data),'highs_mip');m=b.audit(data,r,'highs_mip')
    assert not m['quality_complete'] and m['comparable_value'] is None
    assert m['solver_error_count']==3 and m['rejected_count']==0
    assert m['attempted_batches']==3 and m['batch_solve_quantiles_ms'] is not None
    assert all(x['accepted_value'] is None for x in r['batches'])


def test_uncertified_incumbent_no_horizon_certificate(monkeypatch):
    data,_=fixture()
    def limited(a,**kwargs):
        return SolveResult('highs_mip','feasible_uncertified',(-1,)*len(a.value),0.,a.capacity.copy(),0.,False,'injected')
    monkeypatch.setattr(rolling,'solve',limited)
    r=rolling.simulate(*b.unpack(data),'highs_mip');m=b.audit(data,r,'highs_mip')
    assert m['quality_complete'] and m['comparable_value']==0 and m['per_batch_optimal_count']==0


@pytest.mark.parametrize('case',list(b.CASES))
def test_fixed_workload_hash_loads_and_families(case):
    data=b.generate(case,11);info=b.workload_info(data)
    assert b.digest(data)==b.digest(b.generate(case,11)) and b.digest(data)!=b.digest(b.generate(case,29))
    assert b.CASES[case]<=info['offered_resource_time_load'][0]<b.CASES[case]+.02
    assert len(info['duration_counts'])==4
    cap,jobs,e=b.unpack(data)
    assert np.isfinite(cap).all() and len({j.value for j in jobs})>1
    assert not e[np.array(data['families'])=='CPU'].any()
    if case=='fragmentation':
        for t in range(data['T']):
            assert [j.demand[0]>=4 for j in jobs if j.arrival==t]==sorted(j.demand[0]>=4 for j in jobs if j.arrival==t)


def test_real_six_methods_and_input_hash():
    data,_=fixture()
    for method in METHODS:
        row=b.measure(data,method,.1)
        assert row['metrics']['oracle']=='PASS' and row['input_unchanged']
        assert row['output_sha256']==b.digest(row['report'])
        assert row['simulate_plus_audit_ms']==row['simulate_ms']+row['audit_hash_ms']


def test_empty_and_zero_resource_work_explicit():
    data=b.pack(np.array([[1,4,16]],float),[],np.empty((1,0),bool),['A'],[],b.OMEGA,5,3600,case='empty',seed=11,target=None)
    row=b.measure(data,'first_fit',.1);m=row['metrics']
    assert m['batch_solve_quantiles_ms'] is None and m['attempted_batches']==0
    assert m['admission_rate'] is None and m['window_gpu_hours']==0
