"""PROBE_ONLY physical-node PGFP marginal, not a scheduler or production PGFP.

Strict binary64 comparisons, ZERO resource slack (unlike existing solver tolerance).
GPU coordinates must be whole and <=2**53; float32 inputs are rejected. Invalid
scores have a separate authoritative mask and serialize as null. Each chosen node
is an independent singleton choice, not a simultaneous placement of all tasks.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


def validate_inputs(R, D, Q, weights, E, A):
    arrays = []
    for name, value in [('R', R), ('D', D), ('Q', Q), ('weights', weights)]:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or (raw.dtype.kind == 'f' and raw.dtype.itemsize != 8):
            raise ValueError(f'{name}: require real float64/integer input, not float32/bool')
        if any(isinstance(v, (int, np.integer)) and abs(int(v)) > 2**53
               for v in np.asarray(value, dtype=object).flat):
            raise ValueError(f'{name}: integers above 2**53 are not accepted')
        x = raw.astype(np.float64, copy=True)
        if not np.all(np.isfinite(x)) or np.any(x < 0):
            raise ValueError(f'{name}: require finite nonnegative numbers')
        if name != 'weights':
            if x.ndim != 2 or x.shape[1] != 3:
                raise ValueError(f'{name}: expected (count,3)')
            if np.any(raw[:, 0] > 2**53) or np.any(x[:, 0] != np.floor(x[:, 0])):
                raise ValueError(f'{name}: whole GPU counts <=2**53 required')
        arrays.append(x)
    r, d, q, w = arrays
    if w.shape != (len(q),):
        raise ValueError('weights: expected (future_types,)')
    try:
        math.fsum(w)
    except OverflowError as exc:
        raise ValueError('weights: finite total required') from exc
    masks = []
    for name, value, shape in [('E', E, (len(r), len(q))), ('A', A, (len(r), len(d)))]:
        mask = np.asarray(value)
        if mask.shape != shape or mask.dtype != np.dtype(bool):
            raise ValueError(f'{name}: expected boolean {shape}')
        masks.append(mask.copy())
    return r, d, q, w, *masks


def scalar_oracle(R, D, Q, weights, E, A):
    """Independent nested scalar comparisons and math.fsum of signed changes."""
    r, d, q, w, e, a = validate_inputs(R, D, Q, weights, E, A)
    scores = np.zeros((len(r), len(d)))
    valid = np.zeros(scores.shape, bool)
    for i in range(len(r)):
        for j in range(len(d)):
            valid[i, j] = bool(a[i, j] and all(r[i, k] >= d[j, k] for k in range(3)))
            if not valid[i, j]:
                continue
            changes = []
            for h in range(len(q)):
                if e[i, h]:
                    before = all(float(r[i, k]) >= float(q[h, k]) for k in range(3))
                    after = all(float(r[i, k])-float(d[j, k]) >= float(q[h, k]) for k in range(3))
                    changes.append(float(w[h])*(int(after)-int(before)))
            scores[i, j] = math.fsum(changes)
    return scores, valid


def numpy_marginal(R, D, Q, weights, E, A, *, chunk=32):
    r, d, q, w, e, a = validate_inputs(R, D, Q, weights, E, A)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
        raise ValueError('chunk: positive integer required')
    scores = np.zeros((len(r), len(d)))
    valid = a & np.all(r[:, None, :] >= d[None, :, :], axis=2)
    before = np.all(r[:, None, :] >= q[None, :, :], axis=2) & e
    for start in range(0, len(d), chunk):
        stop = min(len(d), start+chunk)
        left = r[:, None, :]-d[None, start:stop, :]
        after = np.all(left[:, :, None, :] >= q[None, None, :, :], axis=3)
        lost = before[:, None, :] & ~after
        scores[:, start:stop] = -np.sum(lost*w[None, None, :], axis=2)
    scores[~valid] = 0
    return scores, valid


def torch_marginal(R, D, Q, weights, E, A, *, device='cpu', chunk=32):
    """Optional import; each call includes host copies, CUDA sync and return copies."""
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('device: expected cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA_UNAVAILABLE')
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
        raise ValueError('chunk: positive integer required')
    host = validate_inputs(R, D, Q, weights, E, A)
    r, d, q, w, e, a = [torch.tensor(x, device=device) for x in host]
    scores = torch.zeros((len(r), len(d)), dtype=torch.float64, device=device)
    valid = a & torch.all(r[:, None, :] >= d[None, :, :], dim=2)
    before = torch.all(r[:, None, :] >= q[None, :, :], dim=2) & e
    for start in range(0, len(d), chunk):
        stop = min(len(d), start+chunk)
        after = torch.all((r[:, None, :]-d[None, start:stop, :])[:, :, None, :]
                          >= q[None, None, :, :], dim=3)
        scores[:, start:stop] = -torch.sum((before[:, None, :] & ~after)*w[None, None, :], dim=2)
    scores = torch.where(valid, scores, 0.)
    if device == 'cuda':
        torch.cuda.synchronize()
    return scores.cpu().numpy().copy(), valid.cpu().numpy().copy()


def choose_nodes(inputs, scores, valid):
    """Largest marginal, then smallest node index; validate each singleton physically."""
    r, d, _, _, _, a = validate_inputs(*inputs)
    physical = a & np.all(r[:, None, :] >= d[None, :, :], axis=2)
    if (scores.shape != physical.shape or valid.shape != physical.shape
            or valid.dtype != np.dtype(bool) or not np.array_equal(valid, physical)
            or not np.all(np.isfinite(scores)) or np.any(scores[valid] > 0)):
        raise ValueError('score/mask violates physical or nonpositive-marginal contract')
    choices = []
    for j in range(len(d)):
        nodes = np.flatnonzero(valid[:, j])
        i = int(nodes[np.argmax(scores[nodes, j])]) if len(nodes) else None
        if i is not None and not (a[i, j] and np.all(r[i] >= d[j])):
            raise ValueError('invalid physical singleton choice')
        choices.append(i)
    return choices


def json_scores(scores, valid):
    return [[float(scores[i, j]) if valid[i, j] else None for j in range(scores.shape[1])]
            for i in range(scores.shape[0])]


def fixed_inputs():
    rng = np.random.default_rng(11)
    def resources(count, gpu, cpu, ram):
        return np.column_stack((rng.integers(0, gpu, count),
                                rng.integers(0, cpu, count)/4,
                                rng.integers(0, ram, count)/4)).astype(np.float64)
    return (resources(128, 9, 257, 1025), resources(256, 6, 129, 513),
            resources(16, 6, 65, 257), rng.integers(0, 9, 16).astype(float)/4,
            rng.random((128, 16)) < .6, rng.random((128, 256)) < .7)


def fixture_sha256(inputs):
    h = hashlib.sha256()
    for x in inputs:
        h.update(str((x.shape, x.dtype.str)).encode())
        h.update(x.tobytes(order='C'))
    return h.hexdigest()


def run_probe(*, cuda=False):
    """One size/seed; 1 warmup + 3 measured full calls/backend, no compilation."""
    inputs = fixed_inputs()
    digest = fixture_sha256(inputs)
    backends = [('scalar_numpy', scalar_oracle), ('vector_numpy', numpy_marginal)]
    environment = dict(python=sys.version.split()[0], numpy=np.__version__)
    gpu = dict(status='GPU_NOT_RUN', reason='CUDA not requested in this CPU batch')
    try:
        import torch
    except ImportError:
        environment['torch'] = None
        if cuda:
            gpu['reason'] = 'TORCH_NOT_INSTALLED'
    else:
        environment.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), cuda=torch.version.cuda)
        backends.append(('torch_cpu', torch_marginal))
        if cuda and torch.cuda.is_available():
            gpu = dict(status='DEVICE_OBSERVED', name=torch.cuda.get_device_name(0))
            backends.append(('torch_cuda', lambda *a: torch_marginal(*a, device='cuda')))
        elif cuda:
            gpu['reason'] = 'CUDA_UNAVAILABLE'
    reference = None
    records = []
    for name, function in backends:
        times = []
        for repetition in range(4):
            start = time.perf_counter()
            scores, valid = function(*inputs)
            choices = choose_nodes(inputs, scores, valid)
            elapsed = (time.perf_counter()-start)*1000
            if reference is None:
                reference = scores.copy(), valid.copy(), choices.copy()
            error = float(np.max(np.abs(scores-reference[0])))
            if error > 1e-10 or not np.array_equal(valid, reference[1]) or choices != reference[2]:
                raise ValueError(f'{name}: independent oracle/choice mismatch')
            if repetition == 0:
                warmup = elapsed
            else:
                times.append(elapsed)
        records.append(dict(backend=name, max_abs_error=error, mask_equal=True,
                            choices_equal=True, valid_candidates=int(valid.sum()),
                            warmup_ms=warmup, measured_end_to_end_ms=times,
                            median_ms=statistics.median(times), compilation_ms=0,
                            timing='validation + copies/transfers + compute + host return + validated choice; CUDA synchronized'))
    assert fixture_sha256(inputs) == digest
    return dict(status='PASS', mode='PROBE_ONLY', source_sha=os.environ.get('SAE_CANDIDATE_SHA'),
                fixture_sha256=digest, seed=11, shapes=[list(x.shape) for x in inputs],
                dtype='float64', resource_slack=0, score_atol=1e-10,
                chunk=32, warmups=1, measured_calls=3, environment=environment,
                gpu=gpu, records=records,
                scope='independent within-node proxy singleton choices; not production PGFP/scheduler/forecast equivalence')


if __name__ == '__main__':
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)  # This probe process only; never changed on import.
    result = run_probe(cuda='--cuda' in sys.argv or 'SAE_CANDIDATE_SHA' in os.environ)
    if os.environ.get('SAE_CANDIDATE_SHA'):
        if result['gpu']['status'] != 'DEVICE_OBSERVED':
            raise RuntimeError('GPU_NOT_RUN: remote CUDA probe did not execute')
        raw = (json.dumps(result, allow_nan=False, indent=2)+'\n').encode()
        path = Path('/content/gpu-probe.json')
        path.write_bytes(raw)
        print('ARTIFACT', path.name, len(raw), hashlib.sha256(raw).hexdigest())
    print(json.dumps(result, allow_nan=False))
