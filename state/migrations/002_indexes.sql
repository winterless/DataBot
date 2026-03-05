-- Helpful indexes for run recovery and idempotency lookup.
CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_tasks_step_name ON tasks(step_name);
CREATE INDEX IF NOT EXISTS idx_artifacts_task_id ON artifacts(task_id);
