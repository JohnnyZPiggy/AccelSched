import numpy as np
import pytest
from probes import pgfp_physical_profile as p
from probes import pgfp_gpu_scoring as p6
from probes import pgfp_dual_stability as g

@pytest.mark.parametrize('case',['heterogeneous','cpu_tight','ram_eligibility'])
def test_fixed_fixture_equivalence_and_stage_counts(case):
    a=p.p9.fixture(case);lp=p.pg.shape_lp(a);base=p.pg._replay(a,lp)
    for vector in (False,True):
        out,stats=p.replay(a,lp,vector=vector,instrument=True);assert out==base;p.checked(a,out)
        assert stats['counts']['ledger_scalars']==2*a.capacity.size*len(a.value)
        assert abs(sum(stats['stages'].values())+stats['accounting_and_instrumentation_wall']-stats['total_wall'])<1e-9
        assert stats['counts']['marginal_calls']==sum(len(s['candidates']) for q in out['ledger_steps'] for s in q['attempts'])

@pytest.mark.parametrize('a',[p6.edge_case(),g.p4b.fixed_cases()[0][1],g.scenarios()['A_analytic_split']])
def test_tie_epsilon_ineligible_and_competing_jobs(a):
    lp=p.pg.shape_lp(a);assert p.replay(a,lp,vector=True)==p.pg._replay(a,lp)

def test_experiment_actual_cpu_profile_and_bound():
    out=p.experiment('heterogeneous',device={'cuda':False});assert out['baseline']==out['variant'] and len(out['pairs'])==3
    assert out['baseline_profile']['functions'] and out['original_greedy_profile']['functions']
    assert out['shadow']['accepted']==all(out['shadow']['comparison'].values())
    with pytest.raises(ValueError):p.experiment('p5_10k',device={'cuda':False})
