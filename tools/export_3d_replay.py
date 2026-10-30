"""Offline synthetic visual fixture; production scheduling APIs remain unchanged."""
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rolling import Job, simulate

METHODS = ('first_fit', 'classic_best_fit', 'pgfp')


def fixture():
    nodes = [dict(id=f'N{i+1:02}', family=f, capacity=c) for i, (f, c) in enumerate([
        ('A', [4, 32, 128]), ('B', [8, 64, 256]), ('A', [4, 32, 128]),
        ('B', [8, 64, 256]), ('A', [2, 24, 96]), ('B', [4, 48, 192]),
        ('CPU', [0, 64, 256]), ('CPU', [0, 32, 128])])]
    rng = np.random.default_rng(11)
    demands = [(1, 8, 32), (2, 16, 64), (4, 24, 96), (0, 16, 64), (8, 48, 192)]
    jobs, families = [], []
    for i in range(32):
        demand = demands[int(rng.integers(len(demands)))]
        families.append('C' if i == 31 else ('any' if demand[0] == 0 else ['A', 'B'][int(rng.integers(2))]))
        jobs.append(Job(f'J{i+1:02}', (i // 4) * 6, int(rng.choice([6, 12, 18])), demand, int(rng.integers(1, 21))))
    eligible = np.array([[f == 'any' or n['family'] == f for f in families] for n in nodes], dtype=bool)
    return nodes, jobs, families, eligible


def export():
    nodes, jobs, families, eligible = fixture()
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    return dict(schema_version=1, synthetic=True, label='Synthetic cluster • real AccelSched decisions',
                source_sha=sha, source_dirty=dirty, seed=11, time_limit=2.,
                resource_units=['whole GPUs', 'CPU cores', 'host RAM GiB'], nodes=nodes,
                jobs=[dict(**asdict(j), family=f) for j, f in zip(jobs, families)],
                eligible=eligible.tolist(),
                reports={m: simulate([n['capacity'] for n in nodes], jobs, eligible, m, seed=11, time_limit=2.) for m in METHODS})


if __name__ == '__main__':
    path = ROOT / 'demos/accelsched-3d/public/data/replay.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = export()
    path.write_text(json.dumps(bundle, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    print(f"source={bundle['source_sha']} dirty={bundle['source_dirty']} seed=11 time_limit=2.0")
    for method, report in bundle['reports'].items():
        print(method, {k: report[k] for k in ('accepted_count', 'rejected_count', 'solver_error_count')})
    print(path)
