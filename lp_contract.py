"""Canonical sparse ranged LP, original min convention and owned read-only data."""
from dataclasses import dataclass,field
import hashlib
import numpy as np
from scipy.sparse import issparse,csr_matrix


def _vector(value,size,name,bound=False):
    a=np.asarray(value)
    if a.shape!=(size,) or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a):raise ValueError(name+' must be a real vector with matching shape')
    a=np.array(a,dtype=np.float64,copy=True)
    if np.isnan(a).any() or (not bound and not np.isfinite(a).all()):raise ValueError(name+' has invalid coefficients')
    a.flags.writeable=False;return a


@dataclass(frozen=True,eq=False)
class CanonicalLP:
    A: object
    c: np.ndarray
    row_lo: np.ndarray
    row_hi: np.ndarray
    var_lo: np.ndarray
    var_hi: np.ndarray
    c0: float=0.
    source_hash: str=field(init=False)

    def __post_init__(self):
        if not issparse(self.A) or self.A.format!='csr' or np.iscomplexobj(self.A.data) or not np.issubdtype(self.A.data.dtype,np.number):raise ValueError('A must be a real CSR matrix')
        owned=self.A.copy();owned.check_format(full_check=True)
        A=csr_matrix(owned,dtype=np.float64,copy=True);A.sum_duplicates();A.sort_indices()
        if len(A.shape)!=2 or A.shape[1]<1 or not np.isfinite(A.data).all():raise ValueError('invalid CSR coefficients/shape')
        if isinstance(self.c0,(bool,np.bool_)) or not isinstance(self.c0,(int,float,np.number)) or np.iscomplexobj(self.c0) or not np.isfinite(self.c0):raise ValueError('c0 must be finite real')
        object.__setattr__(self,'A',A);object.__setattr__(self,'c0',float(self.c0));m,n=A.shape
        for name,size in (('c',n),('row_lo',m),('row_hi',m),('var_lo',n),('var_hi',n)):
            object.__setattr__(self,name,_vector(getattr(self,name),size,name,name!='c'))
        for lo,hi in ((self.row_lo,self.row_hi),(self.var_lo,self.var_hi)):
            if np.isposinf(lo).any() or np.isneginf(hi).any() or np.any(lo>hi):raise ValueError('invalid bound directions/order')
        for a in (A.data,A.indices,A.indptr):a.flags.writeable=False
        object.__setattr__(self,'source_hash',self.digest())

    def digest(self):
        h=hashlib.sha256()
        for name,a in [('A.data',self.A.data),('A.indices',self.A.indices),('A.indptr',self.A.indptr),('A.shape',np.array(self.A.shape,np.int64)),('c',self.c),('row_lo',self.row_lo),('row_hi',self.row_hi),('var_lo',self.var_lo),('var_hi',self.var_hi),('c0',np.array([self.c0]))]:
            h.update(name.encode());h.update(str((a.dtype.str,a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def assert_intact(self):
        if self.digest()!=self.source_hash:raise ValueError('canonical original model was mutated')
