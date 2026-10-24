"""Offline P2 reporting over one existing P0/P1 comparison, without re-solving."""
import argparse
from copy import deepcopy
import json
import sys

from examples import capacity_whatif, time_window_capacity
from examples.planner_input import InputError, load_input


def _count(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{field}: expected a nonnegative integer')
    return value


def summarize_report(raw_report, mode):
    """Return a detached JSON report; malformed/nonfinite source reports fail clearly."""
    if mode not in ('static', 'dynamic'):
        raise ValueError('mode: expected static or dynamic')
    try:
        json.dumps(raw_report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('source report must contain finite JSON values') from exc
    if not isinstance(raw_report, dict) or not isinstance(raw_report.get('rows'), list):
        raise ValueError('rows: expected a nonempty scenario array')
    rows = raw_report['rows']
    if not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError('rows: expected nonnull scenario objects')
    names = [r.get('scenario') for r in rows]
    if (any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names) or names.count('Baseline') != 1):
        raise ValueError('scenarios: unique names and one named Baseline required')
    if mode == 'dynamic':
        _count(raw_report.get('T'), 'T')
        if raw_report['T'] == 0:
            raise ValueError('T: expected a positive horizon')
    scenarios = []
    for r in rows:
        if not isinstance(r.get('added_nodes'), list):
            raise ValueError('added_nodes: expected an array')
        if mode == 'static':
            valid = (r.get('valid') is True and r.get('status') in
                     ('optimal', 'feasible', 'feasible_uncertified'))
        else:
            errors = _count(r.get('solver_error_count'), 'solver_error_count')
            valid = errors == 0 and r.get('comparison') == 'COMPARABLE'
        count, value = r.get('accepted_count'), r.get('accepted_value')
        if valid:
            _count(count, 'accepted_count')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError('accepted_value: expected a nonnegative number')
        scenario = dict(name=r['scenario'], added_nodes=r['added_nodes'],
                        accepted_count=count if valid else None,
                        accepted_value=value if valid else None, valid=valid)
        if mode == 'static':
            scenario.update(status=r.get('status'), error=r.get('error'),
                            certified=valid and r.get('certified') is True
                            and r.get('solver_optimal') is True and r['status'] == 'optimal',
                            validation_error=r.get('validation_error'))
            for field in ('accepted_ids', 'unplaced_ids', 'per_node_residual',
                          'total_residual', 'utilization'):
                scenario[field] = r.get(field) if valid else None
        else:
            for field in ('rejected_count', 'solver_error_count', 'window_resource_time',
                          'utilization', 'batch_status_counts', 'certified_batch_count', 'batches'):
                if field not in r:
                    raise ValueError(f'{field}: missing dynamic evidence')
                scenario[field] = r[field]
            for field in ('completed_by_T', 'running_at_T'):
                if not isinstance(r.get(field), list):
                    raise ValueError(f'{field}: expected an array')
                scenario[field+'_count'] = len(r[field])
            pressure = r.get('pressure')
            if not isinstance(pressure, dict):
                raise ValueError('pressure: expected reference results or an empty object')
            scenario.update(pressure=pressure,
                            pressure_status='AVAILABLE' if pressure else 'UNAVAILABLE',
                            certification_scope='arrival batches only; no horizon certificate')
        scenarios.append(scenario)
    comparable = all(r['valid'] for r in scenarios)
    base = next(r for r in scenarios if r['name'] == 'Baseline')
    certified = mode == 'static' and all(r['certified'] for r in scenarios)
    comparison = ('static-model optimal comparison' if certified else
                  'observed static candidate comparison') if mode == 'static' else 'observed simulation comparison'
    for r in scenarios:
        r.update(delta_accepted=r['accepted_count']-base['accepted_count'] if comparable else None,
                 delta_value=r['accepted_value']-base['accepted_value'] if comparable else None,
                 comparison=comparison if comparable else 'NOT COMPARABLE')
    leaders = []
    if comparable:
        maximum = max(r['accepted_count'] for r in scenarios)
        leaders = [r['name'] for r in scenarios if r['accepted_count'] == maximum]
    report = dict(report_version=1, mode=mode, input=raw_report.get('input'),
                  baseline='Baseline', scenarios=scenarios,
                  comparison_status='COMPARABLE' if comparable else 'NOT COMPARABLE',
                  leader=leaders[0] if len(leaders) == 1 else None,
                  tied_names=leaders if len(leaders) > 1 else [],
                  resource_units=['GPU-count', 'CPU-cores', 'host-RAM-GiB'])
    if mode == 'dynamic':
        report.update(horizon=raw_report['T'], resource_time_units='resource units multiplied by time')
    json.dumps(report, allow_nan=False)
    return deepcopy(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Shared JSON Input v1 file')
    parser.add_argument('--mode', required=True, choices=('static', 'dynamic'))
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    args = parser.parse_args(argv)
    try:
        config = load_input(args.input, args.mode)
    except InputError as exc:
        print(f'INPUT_INVALID: {exc}', file=sys.stderr)
        return 2
    planner = capacity_whatif if args.mode == 'static' else time_window_capacity
    try:
        report = summarize_report(planner.compare(config), args.mode)
    except (TypeError, ValueError) as exc:
        print(f'REPORT_INVALID: {exc}', file=sys.stderr)
        return 1
    if args.format == 'json':
        print(json.dumps(report, allow_nan=False))
    else:
        print(f"P2 {args.mode} | {report['comparison_status']} | Baseline: {report['baseline']}")
        print('Scenario | Accepted/value | Delta accepted/value | Comparison')
        for r in report['scenarios']:
            print(f"{r['name']} | {r['accepted_count']}/{r['accepted_value']} | "
                  f"{r['delta_accepted']}/{r['delta_value']} | {r['comparison']}")
        print('Largest accepted count: ' + (report['leader'] or
              ('tie: '+', '.join(report['tied_names']) if report['tied_names'] else 'unavailable')))
    return 0 if report['comparison_status'] == 'COMPARABLE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
