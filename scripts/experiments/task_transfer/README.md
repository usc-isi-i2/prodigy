# COVID Cross-Task Transfer Smoke Experiments

This folder contains the first-pass COVID-only task-transfer setup for:

- `nm`: neighbor matching
- `cl`: contrastive same-node view matching
- `fp`: masked node feature prediction

Defaults target Tucker paths under `/dataMeR1`. Override `COVID_ROOT`,
`GRAPH_FILENAME`, `DEVICE`, or pass extra CLI args to the runner when needed.

Dry-run the commands:

```bash
DRY_RUN=1 bash scripts/experiments/task_transfer/run_covid_smokes_tucker.sh
```

Run one task:

```bash
bash scripts/experiments/task_transfer/train_covid_task_tucker.sh cl
```
