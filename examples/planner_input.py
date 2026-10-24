"""Strict, bounded Shared JSON Input v1; no solver calls or global mutation."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from examples.capacity_whatif import Node, Task
from rolling import Job


class InputError(ValueError):
    pass


class _Pairs(list):
    """Retain object pairs until their full path is known (duplicate-key errors)."""


def _fail(path, message):
    raise InputError(f'{path or "$"}: {message}')


def _path(parent, key):
    return f'{parent}.{key}' if parent else key


def _plain(value, path=''):
    if isinstance(value, _Pairs):
        result = {}
        for key, child in value:
            at = _path(path, key)
            if key in result:
                _fail(at, 'duplicate JSON key')
            result[key] = _plain(child, at)
        return result
    if isinstance(value, list):
        return [_plain(child, f'{path}[{i}]') for i, child in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        _fail(path, 'NaN/Infinity and numeric overflow are not valid JSON input')
    return value


def _object(value, path, required, optional=()):
    if not isinstance(value, dict):
        _fail(path, 'expected an object')
    for key in required:
        if key not in value:
            _fail(_path(path, key), 'required field missing')
    for key in value:
        if key not in required and key not in optional:
            _fail(_path(path, key), 'unexpected field')
    return value


def _array(value, path, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        _fail(path, f'expected an array with {minimum}..{maximum} entries')
    return value


def _name(value, path):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        _fail(path, 'expected a nonblank string of at most 128 characters')
    return value  # Exact, case-sensitive; do not trim/normalize identifiers.


def _integer(value, path, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _fail(path, f'expected an integer in [{minimum},{maximum}] (not bool)')
    return value


def _number(value, path, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= maximum):
        _fail(path, f'expected a finite number in [0,{maximum}] (not bool)')
    return value


def _vector(value, path, demand=False):
    value = _array(value, path, 3, 3)
    result = tuple(_number(x, f'{path}[{i}]', 1024 if i == 0 else 1_000_000)
                   for i, x in enumerate(value))
    if result[0] != int(result[0]):
        _fail(f'{path}[0]', 'GPU count must be whole')
    if demand and not any(result):
        _fail(path, 'request must use at least one resource')
    return result


def _nodes(values, path, minimum):
    result, seen = [], set()
    for i, value in enumerate(_array(values, path, minimum, 32)):
        at = f'{path}[{i}]'
        value = _object(value, at, ('id', 'family', 'capacity'))
        id = _name(value['id'], at+'.id')
        if id in seen:
            _fail(at+'.id', 'duplicate node id')
        seen.add(id)
        result.append(Node(id, _name(value['family'], at+'.family'),
                           _vector(value['capacity'], at+'.capacity')))
    return tuple(result)


def _repo():
    root = str(Path(__file__).resolve().parents[1])
    try:
        head = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=2, check=False)
        if head.returncode or not re.fullmatch(r'[0-9a-f]{40}', head.stdout.strip()):
            return dict(repo_commit_sha=None, repo_dirty=None)
        state = subprocess.run(['git', '-C', root, 'status', '--porcelain'], capture_output=True,
                               text=True, timeout=2, check=False)
        return dict(repo_commit_sha=head.stdout.strip(),
                    repo_dirty=bool(state.stdout) if state.returncode == 0 else None)
    except (OSError, subprocess.TimeoutExpired):
        return dict(repo_commit_sha=None, repo_dirty=None)


def load_input(path, expected_mode):
    """Read UTF-8 JSON and validate everything before adapting or solving."""
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(1_048_577)
    except OSError as exc:
        _fail('input', str(exc))
    if len(raw) > 1_048_576:
        _fail('input', 'file exceeds 1 MiB')
    try:
        doc = _plain(json.loads(raw.decode('utf-8'), object_pairs_hook=_Pairs))
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, InputError):
            raise
        _fail('$', f'invalid UTF-8 JSON: {exc}')
    doc = _object(doc, '', ('schema_version', 'mode', 'nodes', 'jobs', 'scenarios'),
                  ('solver', 'horizon', 'pressure_references'))
    _integer(doc['schema_version'], 'schema_version', 1, 1)
    mode = doc['mode']
    if mode not in ('static', 'dynamic') or mode != expected_mode:
        _fail('mode', f'expected {expected_mode!r} for this planner')
    if mode == 'static':
        for key in ('horizon', 'pressure_references'):
            if key in doc:
                _fail(key, 'dynamic-only field')
    horizon = _integer(doc.get('horizon'), 'horizon', 1, 1_000_000) if mode == 'dynamic' else None
    nodes = _nodes(doc['nodes'], 'nodes', 1)
    jobs, requests, job_ids = [], [], set()
    for i, value in enumerate(_array(doc['jobs'], 'jobs', 1, 64)):
        at = f'jobs[{i}]'
        required = ('id', 'family', 'demand') + (('arrival', 'duration') if mode == 'dynamic' else ())
        value = _object(value, at, required, ('value',))
        id = _name(value['id'], at+'.id')
        if id in job_ids:
            _fail(at+'.id', 'duplicate job id')
        job_ids.add(id)
        family = _name(value['family'], at+'.family')
        demand = _vector(value['demand'], at+'.demand', demand=True)
        utility = _number(value.get('value', 1), at+'.value', 1_000_000_000)
        if mode == 'static':
            jobs.append(Task(id, family, demand, utility))
        else:
            arrival = _integer(value['arrival'], at+'.arrival', 0, horizon-1)
            duration = _integer(value['duration'], at+'.duration', 1, 1_000_000)
            jobs.append(Job(id, arrival, duration, demand, utility))
        requests.append(family)
    scenarios, names, baseline_count = [], set(), 0
    for i, value in enumerate(_array(doc['scenarios'], 'scenarios', 2, 8)):
        at = f'scenarios[{i}]'
        value = _object(value, at, ('name', 'add_nodes'))
        name = _name(value['name'], at+'.name')
        if name in names:
            _fail(at+'.name', 'duplicate scenario name')
        names.add(name)
        added = _nodes(value['add_nodes'], at+'.add_nodes', 0)
        if name == 'Baseline':
            baseline_count += 1
            if added:
                _fail(at+'.add_nodes', 'Baseline must have zero additions')
        if len(nodes)+len(added) > 32:
            _fail(at+'.add_nodes', 'inventory exceeds 32 nodes')
        for k, node in enumerate(added):
            if node.id in {n.id for n in nodes}:
                _fail(f'{at}.add_nodes[{k}].id', 'collides with baseline node id')
        scenarios.append((name, nodes+added))
    if baseline_count != 1:
        _fail('scenarios', 'exactly one named Baseline is required')
    settings = _object(doc.get('solver', {}), 'solver', (), ('method', 'seed', 'time_limit'))
    method = settings.get('method', 'highs_mip')
    if method != 'highs_mip':
        _fail('solver.method', 'v1 supports highs_mip only')
    seed = _integer(settings.get('seed', 11), 'solver.seed', 0, 2**31-1)
    limit = _number(settings.get('time_limit', .5), 'solver.time_limit', .5)
    if limit <= 0:
        _fail('solver.time_limit', 'must be positive')
    references = {}
    for i, value in enumerate(_array(doc.get('pressure_references', []), 'pressure_references', 0, 16)):
        at = f'pressure_references[{i}]'
        value = _object(value, at, ('family', 'demand'))
        family = _name(value['family'], at+'.family')
        if family in references:
            _fail(at+'.family', 'duplicate reference family')
        references[family] = _vector(value['demand'], at+'.demand', demand=True)
    solver = dict(method=method, seed=seed, time_limit=limit)
    provenance = dict(schema_version=1, mode=mode, raw_input_sha256=hashlib.sha256(raw).hexdigest(),
                      input_order=dict(nodes=[n.id for n in nodes], jobs=[j.id for j in jobs],
                                       scenarios=[name for name, _ in scenarios],
                                       pressure_families=list(references)),
                      solver=solver.copy(), horizon=horizon, **_repo())
    return dict(mode=mode, nodes=nodes, jobs=tuple(jobs), requests=tuple(requests),
                scenarios=tuple(scenarios), horizon=horizon, references=references,
                solver=solver, provenance=provenance)
