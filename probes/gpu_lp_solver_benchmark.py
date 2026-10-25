"""P7B PROBE_ONLY same-host LP benchmark; no production adapter or routing.

Two named profiles, original P5/ranged certificates, complete per-run archives.
Native defaults are not assumed to meet the independent strict 1e-7 contract.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback
import zipfile
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import gpu_lp_solver_probe as p

BACKENDS=('highs','cupdlpx','hpdlp')
PROFILES=('native_default','certification')
LIMIT=10.


def input_hash(m):
    return p.p5.digest(m['A'].data,m['A'].indices,m['A'].indptr,m['c'],m['l'],m['u'],m['lb'],m['ub'],np.array([m['offset']]))


def order(n,profile,rep):
    shift=((0 if n==1000 else 1 if n==10000 else 2)+PROFILES.index(profile)+rep)%3
    return BACKENDS[shift:]+BACKENDS[:shift]


def settings(backend,profile):
    if profile not in PROFILES:raise ValueError('unknown profile')
    if backend=='highs':
        return dict(time_limit=LIMIT,**(dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-10) if profile=='certification' else {}))
    return dict(TimeLimit=LIMIT,OutputFlag=False,**(dict(OptimalityTol=1e-12,FeasibilityTol=1e-12,Presolve=False) if profile=='certification' else {}))


def call(m,original,backend,profile,modules=None,sync=lambda:None):
    start=perf_counter();out=dict(status='FAILED',backend=backend,profile=profile);vectors={}
    try:
        sync();synced=perf_counter();before=input_hash(m);params=settings(backend,profile)
        if backend=='highs':
            configured=perf_counter();result=linprog(m['c'],A_ub=m['A'],b_ub=m['u'],bounds=(0,None),method='highs',options=params);optimized=perf_counter()
            raw=dict(Status=int(result.status),StatusName=result.message,ObjVal=result.fun,iterations=result.nit);optimal=result.status==0
            x,pi,rc=result.x,result.ineqlin.marginals,result.lower.marginals;effective=dict(overrides=params,native_defaults='SciPy HiGHS default method/presolve/accuracy; no hidden overridden tolerances')
        else:
            module=modules[backend];problem=module.Model(objective_vector=m['c'],constraint_matrix=m['A'],constraint_lower_bound=m['l'],constraint_upper_bound=m['u'],variable_lower_bound=m['lb'],variable_upper_bound=m['ub'],objective_constant=m['offset'])
            problem.ModelSense=module.PDLP.MINIMIZE;problem.setParams(**params)
            roundtrip=dict(A=problem.A,c=problem.c,l=problem.constr_lb,u=problem.constr_ub,lb=problem.lb,ub=problem.ub,offset=problem.c0)
            if input_hash(roundtrip)!=before:raise ValueError('native model CSR/bounds/index roundtrip changed')
            effective=dict(problem._params);configured=perf_counter();problem.optimize();optimized=perf_counter()
            raw={k:getattr(problem,k,None) for k in ('Status','StatusName','StatusCode','ObjVal','DualObj','Gap','Runtime','IterCount','PresolveTime','RescalingTime','RelPrimalResidual','RelDualResidual')}
            optimal=(raw['StatusCode'] if raw.get('StatusCode') is not None else raw['Status'])==module.PDLP.OPTIMAL
            x,pi,rc=problem.X,problem.Pi,problem.RC
            if input_hash(roundtrip)!=before:raise ValueError('native input mutated')
        vectors={k:np.array(v,dtype=np.float64,copy=True) for k,v in zip(('x','pi','rc'),(x,pi,rc)) if v is not None}
        sync();extracted=perf_counter()
        if input_hash(m)!=before:raise ValueError('original input mutated')
        check=p.validate(m,'shape', 'OPTIMAL' if optimal else 'NOT_OPTIMAL',vectors.get('x'),vectors.get('pi'),vectors.get('rc'),raw['ObjVal'],original)
        end=perf_counter();out.update(status='PASS' if check['certified'] else 'UNCERTIFIED',raw_solver=raw,overrides=params,effective_parameters=effective,reported_optimal=optimal,certificate=check,input_sha256=before,
            vector_presence={k:k in vectors for k in ('x','pi','rc')},timing=dict(pre_device_sync=synced-start,model_conversion_roundtrip=configured-synced,optimize_hidden_init_presolve_transfer_postsolve=optimized-configured,extraction_sync=extracted-optimized,original_independent_audit=end-extracted,solver_call_total=end-start))
    except Exception:out.update(error=traceback.format_exc(),timing=dict(solver_call_total=perf_counter()-start),vector_presence={k:k in vectors for k in ('x','pi','rc')})
    return out,vectors


def archive(vectors):
    buf=io.BytesIO();np.savez_compressed(buf,**vectors);return buf.getvalue(),{k:dict(shape=list(v.shape),dtype=v.dtype.str,bytes=v.nbytes,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in vectors.items()}


def eligible_100k(records,remaining,ram,free,estimate):
    small=[q for q in records if q['profile']=='certification' and q['n'] in (1000,10000)]
    return len(small)==12 and all(q['status']=='PASS' for q in small) and remaining>=180 and ram>=4*estimate and free>=2*estimate


def run(directory,budget=360.):
    import cupdlpx,hpdlp,cupy as cp,psutil
    directory=Path(directory);directory.mkdir(exist_ok=True);start=perf_counter();records=[];report=dict(status='PASS',mode='PROBE_ONLY',profiles=list(PROFILES),records=records,inputs={},scale_decisions=[],limits=dict(worker_seconds=budget,native_limit=LIMIT,max_n=100000,repetitions=2,chunk_raw_bytes=4*1024**2,max_archive_bytes=128*1024**2),backends=list(BACKENDS))
    modules=dict(cupdlpx=cupdlpx,hpdlp=hpdlp);sync=cp.cuda.runtime.deviceSynchronize
    sync();report['cuda_ready']=dict(free_total_bytes=list(cp.cuda.runtime.memGetInfo()),runtime=cp.cuda.runtime.runtimeGetVersion())
    def persist():
        (directory/'results.json').write_text(json.dumps(p.json_safe(report),separators=(',',':'),allow_nan=False))
    for n,nodes in ((1000,64),(10000,256),(100000,1024)):
        remaining=budget-(perf_counter()-start);ram=psutil.virtual_memory().available;free,total=cp.cuda.runtime.memGetInfo();estimate=p.p5.dimensions(n,nodes)['conservative_working_bytes']
        allow=n<100000 or eligible_100k(records,remaining,ram,free,estimate)
        report['scale_decisions'].append(dict(n=n,run=allow,remaining_seconds=remaining,host_available_bytes=ram,gpu_free_bytes=free,estimated_working_bytes=estimate,reason='OBLIGATORY_SMALL' if n<100000 else 'STRICT_SMALL_CERTIFICATES_AND_MEASURED_BUDGET_RAM_VRAM' if allow else 'NOT_RUN_CERTIFICATION_BUDGET_OR_MEMORY'))
        if not allow:break
        for profile in PROFILES:
            for rep in range(2):
                for backend in order(n,profile,rep):
                    if perf_counter()-start>budget-20:report['status']='PARTIAL_BUDGET';break
                    begin=perf_counter()
                    with p.p5.Memory() as mem:
                        m,original,meta=p.shape(n,nodes);ih=input_hash(m);generated=perf_counter();out,vec=call(m,original,backend,profile,modules,sync)
                        key=f'{n}_{profile}_{rep}_{backend}';packed,vm=archive(vec);(directory/(key+'.npz')).write_bytes(packed);stored=perf_counter()
                    done=perf_counter();out.update(key=key,n=n,nodes=nodes,repetition=rep,label='first' if rep==0 else 'warm',execution_index=len(records),profile=profile,backend=backend,input_sha256=ih,input_metadata=meta,
                        archive_bytes=len(packed),archive_sha256=hashlib.sha256(packed).hexdigest(),vector_metadata=vm,memory=mem.report())
                    out['timing'].update(fixture_csr_canonical_hash=generated-begin,serialize_write=stored-generated-out['timing']['solver_call_total'],sampler_exit=done-stored,source_to_certified_archive_total=done-begin)
                    records.append(out);report['inputs'][str(n)]=dict(meta,input_sha256=ih);persist()
                if report['status']=='PARTIAL_BUDGET':break
            if report['status']=='PARTIAL_BUDGET':break
        if report['status']=='PARTIAL_BUDGET':break
    # Immutable per-record archives avoid charging cumulative compression to later solvers.
    combined=io.BytesIO()
    with zipfile.ZipFile(combined,'w',zipfile.ZIP_STORED) as z:
        for q in records:z.writestr(q['key']+'.npz',(directory/(q['key']+'.npz')).read_bytes())
    raw=combined.getvalue()
    if len(raw)>128*1024**2:raise ValueError('bounded complete vector archive cap exceeded')
    digest=hashlib.sha256(raw).hexdigest();blocks=[raw[i:i+4*1024**2] for i in range(0,len(raw),4*1024**2)];names=[]
    for index,block in enumerate(blocks):
        name=f'vector-{index:03}.json';names.append(name);payload=dict(index=index,total=len(blocks),archive_sha256=digest,archive_bytes=len(raw),block_sha256=hashlib.sha256(block).hexdigest(),base64=base64.b64encode(block).decode())
        (directory/name).write_text(json.dumps(payload,separators=(',',':')))
    report.update(artifacts=names,combined_archive_bytes=len(raw),combined_archive_sha256=digest,worker_probe_seconds=perf_counter()-start,
        certified_fractions={f'{n}_{profile}_{backend}':dict(certified=sum(q['status']=='PASS' for q in records if q['n']==n and q['profile']==profile and q['backend']==backend),attempted=sum(q['n']==n and q['profile']==profile and q['backend']==backend for q in records)) for n in sorted({q['n'] for q in records}) for profile in PROFILES for backend in BACKENDS})
    if any(q['status']!='PASS' for q in records):report['status']='COMPLETED_WITH_UNCERTIFIED'
    persist();return report


def audit(summary_path,directory,output):
    wrapper=json.loads(Path(summary_path).read_bytes());report=wrapper['experiment'];chunks=[]
    for i,meta in enumerate(wrapper['trace_artifacts']):
        raw=(Path(directory)/meta['name']).read_bytes()
        if len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('transport SHA')
        item=json.loads(raw)['data'];block=base64.b64decode(item['base64'],validate=True)
        if item['index']!=i or item['total']!=len(wrapper['trace_artifacts']) or hashlib.sha256(block).hexdigest()!=item['block_sha256']:raise ValueError('chunk sequence/hash')
        chunks.append(block)
    raw=b''.join(chunks)
    if len(raw)!=report['combined_archive_bytes'] or hashlib.sha256(raw).hexdigest()!=report['combined_archive_sha256']:raise ValueError('combined archive SHA')
    (Path(directory)/'complete-vectors.zip').write_bytes(raw);z=zipfile.ZipFile(io.BytesIO(raw));checks=[];models={}
    if set(z.namelist())!={q['key']+'.npz' for q in report['records']}:raise ValueError('record archive coverage')
    for q in report['records']:
        n=q['n']
        if n not in models:models[n]=p.shape(n,q['nodes'])
        m,original,meta=models[n]
        if input_hash(m)!=q['input_sha256'] or meta!=q['input_metadata']:raise ValueError('original model/input identity')
        packed=z.read(q['key']+'.npz')
        if len(packed)!=q['archive_bytes'] or hashlib.sha256(packed).hexdigest()!=q['archive_sha256']:raise ValueError('record archive SHA')
        v=np.load(io.BytesIO(packed),allow_pickle=False)
        if set(v.files)!=set(q['vector_metadata']):raise ValueError('array coverage')
        for field,vm in q['vector_metadata'].items():
            arr=v[field]
            if list(arr.shape)!=vm['shape'] or arr.dtype.str!=vm['dtype'] or arr.nbytes!=vm['bytes'] or hashlib.sha256(arr.tobytes()).hexdigest()!=vm['sha256']:raise ValueError('array bytes')
        check=p.validate(m,'shape','OPTIMAL' if q.get('reported_optimal') else 'NOT_OPTIMAL',v['x'] if 'x' in v else None,v['pi'] if 'pi' in v else None,v['rc'] if 'rc' in v else None,q.get('raw_solver',{}).get('ObjVal'),original)
        if check['certified']!=q.get('certificate',{}).get('certified',False):raise ValueError('remote/local certification disagreement')
        checks.append(dict(key=q['key'],status='PASS' if check['certified'] else 'UNCERTIFIED',certificate=check))
    out=dict(status='PASS_COMPLETE_ORIGINAL_MODEL_AUDIT',project_sha=wrapper['project_sha'],adapter_sha=wrapper['source_sha'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,archive_sha256=report['combined_archive_sha256'])
    Path(output).write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],records=len(checks),certified=sum(q['status']=='PASS' for q in checks))));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir');parser.add_argument('--audit-summary');parser.add_argument('--audit-directory');parser.add_argument('--audit-output');args=parser.parse_args()
    if args.audit_summary:audit(args.audit_summary,args.audit_directory,args.audit_output)
    else:run(args.output_dir)
