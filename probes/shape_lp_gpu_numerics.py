"""P5B PROBE_ONLY: bounded numerical repairs; immutable P5A model/audit policy.

R*A*D, R*b, D*c with x=D*z, Pi=R*Pi_scaled, RC=RC_scaled/D.
All acceptance is against the ORIGINAL CSR and unchanged P5A tolerance.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import shape_lp_gpu_scale as base

REPAIRS=(dict(name='tighter_controls',tolerance=1e-12,scaled=False),
         dict(name='reversible_equilibration',tolerance=1e-12,scaled=True))
P5A_BLOB='375fc518ec718ef4e6ccdea27db49b4d1aa2a434'


def equilibrate(model):
    """Exact binary diagonal factors; no dense matrix and no changed feasible LP."""
    A=model['A'];m,n,r=model['m'],model['n'],model['r']
    # Resource coefficients for each shape/job determine each column maximum.
    maxima=np.ones(m*n)
    for row in range(n,A.shape[0]):
        lo,hi=A.indptr[row:row+2];idx=A.indices[lo:hi]
        maxima[idx]=np.maximum(maxima[idx],np.abs(A.data[lo:hi]))
    column=np.exp2(-np.floor(np.log2(maxima)))
    row=np.exp2(-np.floor(np.log2(np.maximum(1.,np.abs(model['b'])))))
    data=A.data.copy()
    # Chunking avoids an nnz-sized gathered column-factor temporary.
    for lo in range(0,A.nnz,1000000):
        hi=min(A.nnz,lo+1000000);data[lo:hi]*=column[A.indices[lo:hi]]
    B=csr_matrix((data,A.indices.copy(),A.indptr.copy()),shape=A.shape)
    for lo in range(0,A.shape[0],100000):
        hi=min(A.shape[0],lo+100000)
        left,right=B.indptr[lo],B.indptr[hi]
        B.data[left:right]*=np.repeat(row[lo:hi],np.diff(B.indptr[lo:hi+1]))
    scaled=model.copy();scaled.update(A=B,b=row*model['b'],c=column*model['c'])
    return scaled,row,column


def inverse(z,pi,rc,row,column):
    return column*np.asarray(z),row*np.asarray(pi),np.asarray(rc)/column


def diagnostics(model,x,pi):
    A,b=model['A'],model['b'];violation=A@x-b;y=-pi;n=model['n'];r=model['r']
    ids=np.argpartition(violation,-min(5,len(violation)))[-5:]
    ids=sorted(ids,key=lambda i:(-violation[i],int(i)))
    rows=[]
    for i in ids:
        rows.append(dict(row=int(i),kind='job' if i<n else 'shape_resource',
            job=int(i) if i<n else None,shape=None if i<n else int((i-n)//r),
            resource=None if i<n else int((i-n)%r),violation=float(violation[i]),
            rhs=float(b[i]),lhs=float(b[i]+violation[i])))
    norm=lambda z:dict(l1=float(np.linalg.norm(z,1)),l2=float(np.linalg.norm(z)),linf=float(np.max(np.abs(z))))
    return dict(worst_rows=rows,rows_above_fixed_threshold=int(np.sum(violation>base.TOL)),
        primal_norm=norm(x),canonical_dual_norm=norm(y))


def solve(model,backend,config,limit,module=None):
    tick=perf_counter();raw={};x=pi=rc=None
    try:
        used=model;row=column=None
        if config.get('scaled'):used,row,column=equilibrate(model)
        transformed=perf_counter()
        if backend=='highs':
            parameters=dict(time_limit=limit,primal_feasibility_tolerance=1e-9,
                dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-10)
            setup=perf_counter();p=linprog(used['c'],A_ub=used['A'],b_ub=used['b'],bounds=(0,None),
                method='highs-ipm',options=parameters)
            done=perf_counter();raw=dict(Status=int(p.status),StatusName=p.message,ObjVal=p.fun,
                iterations=int(p.nit),crossover_iterations=int(getattr(p,'crossover_nit',0)))
            if p.x is None or p.ineqlin.marginals is None:
                return dict(status='TIMEOUT_NO_CERTIFICATE' if p.status==1 else 'NO_CERTIFICATE',
                    backend=backend,configuration=config,parameters=parameters,raw_solver=raw,
                    method='highs-ipm',timing=dict(total=perf_counter()-tick)),None
            x,pi,rc=p.x,p.ineqlin.marginals,p.lower.marginals;optimal=p.status==0
        else:
            p=module.Model(objective_vector=used['c'],constraint_matrix=used['A'],
                constraint_lower_bound=np.full(len(used['b']),-np.inf),constraint_upper_bound=used['b'],
                variable_lower_bound=np.zeros(len(used['c'])),variable_upper_bound=np.full(len(used['c']),np.inf))
            p.ModelSense=module.PDLP.MINIMIZE
            parameters=dict(TimeLimit=limit,OptimalityTol=config['tolerance'],
                FeasibilityTol=config['tolerance'],Presolve=False,OutputFlag=False)
            p.setParams(**parameters);setup=perf_counter();p.optimize();done=perf_counter()
            raw={k:getattr(p,k,None) for k in ('Status','StatusName','ObjVal','DualObj','Gap','Runtime','IterCount',
                'RelPrimalResidual','RelDualResidual')}
            x,pi,rc=np.asarray(p.X,float),np.asarray(p.Pi,float),np.asarray(p.RC,float)
            optimal=p.Status==module.PDLP.OPTIMAL
        retrieved=perf_counter()
        if row is not None:x,pi,rc=inverse(x,pi,rc,row,column)
        mapped=perf_counter()
        audit=base.certify(model,x,pi,rc,optimal,raw['ObjVal'],raw.get('DualObj'),raw.get('Gap'))
        extra=diagnostics(model,x,pi);end=perf_counter()
        result=dict(status='PASS' if audit['certified'] else 'UNCERTIFIED',backend=backend,
            configuration=config,parameters=parameters,raw_solver=raw,certificate=audit,diagnostics=extra,
            method='highs-ipm' if backend=='highs' else 'cuPDLPx',
            timing=dict(reversible_transform=transformed-tick,model_setup=setup-transformed,
                optimize_including_hidden_transfers=done-setup,retrieval=retrieved-done,
                inverse_mapping=mapped-retrieved,original_certificate_diagnostics=end-mapped,total=end-tick),
            transform=None if row is None else dict(row_min=float(row.min()),row_max=float(row.max()),
                column_min=float(column.min()),column_max=float(column.max()),factor_sha256=base.digest(row,column),
                formulas='A_scaled=R A D; b_scaled=R b; c_scaled=D c; x=D z; Pi=R Pi_scaled; RC=RC_scaled/D'),
            hidden_transfer_seconds=None)
        return result,(x,pi,rc)
    except Exception:
        return dict(status='FAILED',backend=backend,configuration=config,raw_solver=raw,
            error=traceback.format_exc(),timing=dict(total=perf_counter()-tick)),None


def run(directory,budget=320.):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import cupdlpx
    from cupdlpx import PDLP
    cupdlpx.PDLP=PDLP;imported=perf_counter()
    import psutil
    report=dict(status='PARTIAL',mode='PROBE_ONLY',tiers=[],repairs=[],max_repairs=2,
        import_seconds=imported-start,seed=71,p5a_blob=P5A_BLOB,
        policy='GPU first, baseline100K20s; at most2 global repairs after actual uncertified result. '
            'Tight controls40s at100K/60s1M; scaling50s100K/60s1M. CPU highs-ipm30s; '
            'one certified warm repeat only if >=solve limit+15s remains. '
            'One first+one warm per successful config, no more than4 calls/backend/tier.',
        tolerance_policy='Unmodified P5A original-model certify: absolute1e-7; scaled gap/row/column '
            'complementarity1e-7*max(1,|primal|,|dual|); RC complementarity absolute1e-7.',
        planned=[base.dimensions(n,s) for n,s in ((100000,1024),(1000000,4096))],
        gpu_memory_before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip())
    gpu_total,gpu_free=(int(z.strip())*1024**2 for z in report['gpu_memory_before'].split(','))
    def persist():
        report['elapsed_seconds']=perf_counter()-start
        (directory/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    def attempt(model,backend,config,cap,label,case):
        remaining=budget-(perf_counter()-start)
        if remaining<10:return None
        cap=min(cap,remaining-5)
        with base.Memory() as memory:record,vectors=solve(model,backend,config,cap,cupdlpx)
        record.update(label=label,memory=memory.report(),outer_remaining_seconds=remaining,
            from_scratch_total_seconds=case['fixture_seconds']+case['assembly_seconds']+record['timing']['total'])
        case['runs'][backend].append(record)
        # Preserve one failed100K and one successful1M actual vector set, without another solve.
        keep=(case['n']==100000 and backend=='cupdlpx' and record['status']!='PASS' and not report.get('failed100k_vectors')) or (case['n']==1000000 and backend=='cupdlpx' and record['status']=='PASS' and not report.get('successful1m_vectors'))
        if keep and vectors is not None:
            saved=perf_counter();path=directory/(str(case['n'])+'-'+label+'-vectors.npz')
            np.savez_compressed(path,x=vectors[0],pi=vectors[1],rc=vectors[2]);raw=path.read_bytes()
            entry=dict(name=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                vector_sha256=base.digest(*vectors),compression_seconds=perf_counter()-saved,
                scope='Original-model vectors after inverse mapping if any.')
            record['vectors']=entry;record['certificate']['retained_vectors']='Full original-model x/Pi/RC compressed locally; transport availability recorded by driver.'
            report['failed100k_vectors' if case['n']==100000 else 'successful1m_vectors']=entry
        del vectors;persist();return record
    # First observed GPU invocation and empirical Pi-sign calibration.
    calibration=base.solve_one(base.fixtures()['hand_unique'],'cupdlpx',cupdlpx,calibrate=True)
    report['gpu_first_calibration']=calibration;persist()
    if calibration['status']!='PASS':report['status']='BLOCKED_CALIBRATION';persist();return report
    chosen=dict(name='baseline',tolerance=1e-10,scaled=False)
    for n,nodes in ((100000,1024),(1000000,4096)):
        d=base.dimensions(n,nodes);available=psutil.virtual_memory().available
        # Include original+scaled CSR, solver/native copies and audit vectors in conservative bound.
        required=d['conservative_working_bytes']*2
        gpu_required=d['estimated_csr_bytes']*2+8*(8*n*8+(n+24)*5)
        if required>.65*available or gpu_required>.70*gpu_free:
            report['tiers'].append(dict(**d,status='NOT_RUN_GROUNDED_MEMORY_BOUND',host_available=available,
                conservative_host_required=required,gpu_free=gpu_free,conservative_gpu_required=gpu_required));persist();break
        if budget-(perf_counter()-start)<50:
            report['tiers'].append(dict(**d,status='NOT_RUN_EXHAUSTED_OBSERVED_TIME_BUDGET'));persist();break
        tick=perf_counter();a=base.fixture(n,nodes);made=perf_counter();model=base.assemble(a);built=perf_counter()
        A=model['A'];case=dict(**d,fixture_seconds=made-tick,assembly_seconds=built-made,
            fixture_sha256=base.digest(a.capacity,a.demand,a.value,a.shapes),
            lp_sha256=base.digest(A.data,A.indices,A.indptr,model['b'],model['c']),
            csr_bytes=dict(data=A.data.nbytes,indices=A.indices.nbytes,indptr=A.indptr.nbytes),actual_nnz=A.nnz,
            memory_gate=dict(host_available_bytes=available,estimated_host_peak_bytes=required,
                gpu_free_bytes=gpu_free,estimated_gpu_peak_bytes=gpu_required),runs=dict(highs=[],cupdlpx=[]))
        report['tiers'].append(case);persist()
        current=attempt(model,'cupdlpx',chosen,20 if n==100000 else 60,'first',case)
        while current and current['status']!='PASS' and len(report['repairs'])<2 and budget-(perf_counter()-start)>80:
            config=REPAIRS[len(report['repairs'])].copy()
            report['repairs'].append(dict(cycle=len(report['repairs'])+1,n=n,trigger_status=current['status'],
                trigger_certificate=current.get('certificate'),configuration=config))
            current=attempt(model,'cupdlpx',config,(40 if len(report['repairs'])==1 else 50) if n==100000 else 60,
                'repair'+str(len(report['repairs'])),case)
            if current and current['status']=='PASS':chosen=config
        if current and current['status']=='PASS':
            chosen=current['configuration']
            if budget-(perf_counter()-start)>100:
                attempt(model,'cupdlpx',chosen,40 if n==100000 else 60,'warm',case)
        # A100K CPU timeout never blocks the next GPU tier; use IPM based on P5A simplex/auto limit.
        cpu=attempt(model,'highs',dict(name='same_host_highs_ipm',scaled=False),30,'first',case)
        if cpu and cpu['status']=='PASS' and budget-(perf_counter()-start)>150:
            attempt(model,'highs',dict(name='same_host_highs_ipm',scaled=False),30,'warm',case)
        case['status']='PASS' if all(any(z['status']=='PASS' for z in case['runs'][b]) for b in ('highs','cupdlpx')) else 'PARTIAL_UNCERTIFIED'
        valid={b:[z for z in case['runs'][b] if z['status']=='PASS'] for b in ('highs','cupdlpx')}
        case['certified_comparison']=None if not all(valid.values()) else dict(
            cpu_over_gpu_full_path=valid['highs'][-1]['timing']['total']/valid['cupdlpx'][-1]['timing']['total'],
            cpu_label=valid['highs'][-1]['label'],gpu_label=valid['cupdlpx'][-1]['label'],
            objective_difference=abs(valid['highs'][-1]['certificate']['primal_objective']-valid['cupdlpx'][-1]['certificate']['primal_objective']))
        persist();del model,a;gc.collect()
    report['status']='PASS' if len(report['tiers'])==2 and all(c.get('status')=='PASS' for c in report['tiers']) else 'PARTIAL_UNCERTIFIED'
    report['not_attempted_jobs']=[n for n in (100000,1000000) if not any(t['n']==n and 'runs' in t for t in report['tiers'])]
    persist();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();result=run(args.output_dir)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])))
