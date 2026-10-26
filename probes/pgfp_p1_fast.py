"""P1 experimental live-watcher dispatch; original benchmark defaults untouched.

Only for the existing static replay with immutable Shape/family/group metadata.
Lists stay live so queries registered AFTER initialization are still updated.
Scores are never treated as monotone; all original candidate updates remain.
"""
import bench_ultra_pgfp as baseline


class CandidateIndex(baseline.ExactIndex):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        dispatch={};self._node_watchers=[]
        for shape,family,group in zip(self.shapes,self.nf,self.ng):
            key=(int(shape),int(family),int(group))
            if key not in dispatch:
                dispatch[key]=tuple(self.watchers.setdefault((key[0],f,g),[])
                    for f in {-1,key[1]} for g in {-1,key[2]})
            self._node_watchers.append(dispatch[key])

    def commit(self,node,demand):
        old=self.rows[node];d=tuple(map(int,demand));new=tuple(old[k]-d[k] for k in range(3));assert min(new)>=0
        self.rows[node]=new;self.sums[node]=sum(new);self.rest[node]=new
        self.refresh_future(node);self.counters['node_commits']+=1
        for watchers in self._node_watchers[node]:
            for q in watchers:q.update(node)
