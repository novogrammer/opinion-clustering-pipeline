# Project Template

`init-project` が内部で使う `projects/{project_name}/` のひな形。

- `00_raw/` に受領した元データを置く
- `01_processed/` に `responses_normalized.csv` を置く
- `02_screening/` に `screened_responses.csv` を置く
- `questions/{question_id}/` 配下は設問ごとに作る
- `99_logs/` に `pipeline.log`, `raw_to_processed.log`, `screening.log` を残す
