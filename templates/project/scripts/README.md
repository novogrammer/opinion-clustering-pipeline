# Project Scripts Template

案件固有の `00_raw -> 01_processed` 変換スクリプト置き場。公開I/Fではなく、Codex が必要時に内部作業として使う前提。

- 単純な1CSV正規化で足りるなら `scripts/normalize_processed.py` を優先する
- `normalize_responses.py` を出発点にする
- 列対応や結合条件は案件ごとに編集する
- `normalize_responses.py` の最後は共通ヘルパーで `responses_normalized.csv` とログ出力にそろえる
- 出力は `01_processed/responses_normalized.csv` にそろえる
- 実行後は `99_logs/raw_to_processed_mapping.md` と `99_logs/raw_to_processed.log` を更新する
