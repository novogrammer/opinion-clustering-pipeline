# Project Template

`init-project` が内部で使う `projects/{project_name}/` のひな形。

- `00_raw/` に受領した元データを置く
- `01_processed/` に `responses_normalized.csv` を置く
- `02_screening/` に `screened_responses.csv` を置く
- `scripts/` にプロジェクト専用の raw 変換スクリプトを置ける
- `questions/{question_id}/` 配下は設問ごとに作る
- `99_logs/` に `pipeline.log`, `raw_to_processed.log`, `screening.log` を残す
- `99_logs/raw_to_processed_mapping.md` は、標準4列への変換内容を残す必要がある場合だけ使う
