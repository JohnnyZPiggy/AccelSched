import numpy as np
import pytest
from probes import gpu_resident_feasibility as p

@pytest.mark.parametrize('case',['epsilon','tie_competing','safe_disjoint','shared_reads','aidc64'])
def test_reused_cpu_paths_full_ledger_and_compaction(case):
    a=p.fixture(case);r=p.measure(case,'cpu_original_physical',0);v=p.run_physical(a,r['result'],'p10_numpy');assert p.common(r['physical']['output'])==p.common(v['output'])
    cert=p.certificate(a,r['result']);batch=p.c.replay(a,cert,'numpy');assert p.common(batch)==p.common(v['output'])
    proof=p.compact_proof(a,batch);assert proof['verified_against_all_observed_states'] and len(proof['observed_choices'])==len(a.value)
    assert all(p.c.independent(a.eligible,jobs) for jobs in batch['batch']['batch_task_ids'])

def test_compact_tamper_and_10k_guard():
    a=p.fixture('safe_disjoint');r=p.measure('safe_disjoint','cpu_original_physical',0);out=r['physical']['output'];out['ledger_steps'][0]['chosen']=-1
    with pytest.raises(AssertionError):p.compact_proof(a,out)
    with pytest.raises(ValueError):p.fixture('p5_10k')
