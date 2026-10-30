export type Job = {id: string; arrival: number; duration: number; demand: number[]; family: string};
export type Decision = {job_id: string; node: number | null; start: number | null; end: number | null; status: string; solver_status: string};
export type Event = {time: number; accepted: string[]; completed: string[]; rejected: string[]; solver_errors: string[]; running: {job_id: string; node: number; end: number}[]; running_load: number[][]; free: number[][]; batch_index: number | null};
export type Report = {decisions: Decision[]; events: Event[]; batches: {solver_status: string; validation_status: string; error: string | null}[]; accepted_count: number; rejected_count: number; solver_error_count: number};
export type Bundle = {schema_version: number; synthetic: boolean; source_sha: string | null; seed: number; nodes: {id: string; family: string; capacity: number[]}[]; jobs: Job[]; reports: Record<string, Report>};
export function stateAt(report: Report, time: number) {
  return report.events.findLast(e => e.time <= time);
}
