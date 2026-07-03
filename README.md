# Opinion Clustering Pipeline

実装前提:

- Python `3.11`
- 仮想環境は `venv`
- 依存管理は `requirements.txt`

案件ごとの入力データと生成成果物は `projects/` 配下に分けて置く。

推奨構成:

```txt
projects/
  {project_name}/
    00_raw/
    01_processed/
    02_screening/
    questions/
      {question_id}/
        03_embeddings/
        04_clustering/
        05_classification/
        06_review/
    99_logs/
```

例:

```txt
projects/customer_survey_2026q3/
projects/support_voice_nps_wave1/
```

`project_name` はディレクトリ名としてそのまま使う。半角英数字と `_` を基本にし、案件開始時に固定する。

各フォルダは段階を表す。連番を付けることで、案件を開いたときに進行順がそのまま分かる。

- `01_processed` は判断なしの整形結果
- `02_screening` で無回答や分類対象外を判定する
- `03_embeddings` 以降は `questions/{question_id}/` 配下で設問ごとに進める

`02_screening/screened_responses.csv` は、`01_processed` の列を引き継ぎつつ `is_target` と `screening_reason` を追加する想定。

初期運用では `screened_responses.csv` を1本で管理し、`screening_reason` は `target`, `blank`, `non_response`, `symbol_only` の4種類を使う。

`01_processed/responses_normalized.csv` は、CSV / UTF-8 / 1行1回答 / 4列固定の状態になっていれば次段階へ進める。

`03_embeddings` は `02_screening/screened_responses.csv` のうち `is_target = true` の行だけを使う。`embedding_requests.csv` の最小列は `response_id`, `question_id`, `embedding_input_text`。

`projects/` 配下の案件データと生成成果物は、原則 Git 管理しない。管理対象は `docs/`、コード、必要なら匿名化済みサンプルだけに絞る。

共通ドキュメントは `docs/`、今後の共通コードは `src/` または `scripts/` に分ける前提とする。

セットアップ例:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
