# Opinion Clustering Pipeline

自由記述回答の分類パイプラインを整理するためのリポジトリ。

## 環境構築

前提:

- Python `3.11`
- 仮想環境は `venv`
- 依存管理は `requirements.in` + `requirements.txt`

セットアップ:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 依存管理

ルール:

- `requirements.in` は人が編集する最小依存
- `requirements.txt` は固定版の lock ファイル
- `requirements.txt` は `pip-compile` で `requirements.in` から生成する
- `requirements.in` を編集したら `requirements.txt` を再生成する

`requirements.txt` の再生成:

```bash
source .venv/bin/activate
pip-compile requirements.in -o requirements.txt
```

## ディレクトリ構成

案件ごとの入力データと生成成果物は `projects/` 配下に分けて置く。

基本構成:

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
        05_curation/
        06_classification/
    99_logs/
```

- `01_processed` は判断なしの整形結果
- `02_screening` で無回答や分類対象外を文字列ルールで自動判定する
- `02_screening` では文脈解釈をせず、空欄、定型無回答、記号のみを機械的に切り分ける
- `05_curation` で LLM 草案を作り、人がカテゴリマスタを確定する
- `06_classification` でカテゴリマスタと embedding を使ったベクトル近傍分類を行う
- `03_embeddings` 以降は `questions/{question_id}/` 配下で設問ごとに進める

`projects/` 配下の案件データと生成成果物は、原則 Git 管理しない。管理対象は `docs/`、コード、必要なら匿名化済みサンプルだけに絞る。

共通ドキュメントは `docs/`、今後の共通コードは `src/` または `scripts/` に分ける前提とする。

## 公開I/F

公開I/F は `python scripts/<script>.py ...` に固定する。

- 案件作成は `init_project.py`
- 設問作成は `init_question.py`
- 生CSVの標準化は `normalize.py --input ...`
- 以降の工程は `screening.py` から `classification.py` までを順に実行する
- `init_project.py` と `init_question.py` は sample 成果物を複製しない

標準フロー:

```bash
python scripts/init_project.py --project-name your_project_name
python scripts/init_question.py --project-dir projects/your_project_name --question-id Q1
python scripts/normalize.py --input projects/your_project_name/00_raw/source.csv --output projects/your_project_name/01_processed/responses_normalized.csv --mapping-log projects/your_project_name/99_logs/raw_to_processed_mapping.md --run-log projects/your_project_name/99_logs/raw_to_processed.log --response-id-col 回答ID --question-id-col 設問ID --question-text-col 質問文 --answer-text-col 自由回答
python scripts/screening.py --input projects/your_project_name/01_processed/responses_normalized.csv --output projects/your_project_name/02_screening/screened_responses.csv --log projects/your_project_name/99_logs/screening.log
python scripts/embeddings.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --output-dir projects/your_project_name/questions/Q1/03_embeddings
python scripts/clustering.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --embeddings projects/your_project_name/questions/Q1/03_embeddings/embeddings.npy --output-dir projects/your_project_name/questions/Q1/04_clustering
python scripts/curation.py --input projects/your_project_name/02_screening/screened_responses.csv --clusters projects/your_project_name/questions/Q1/04_clustering/clusters.csv --question-id Q1 --draft-model gpt-4.1-mini --output-dir projects/your_project_name/questions/Q1/05_curation
python scripts/classification.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --embeddings projects/your_project_name/questions/Q1/03_embeddings/embeddings.npy --category-master projects/your_project_name/questions/Q1/05_curation/category_master.csv --output-dir projects/your_project_name/questions/Q1/06_classification
```

`normalize.py` の標準機能は、1 CSV を標準4列へ写像する単純な列対応までとする。  
`00_raw -> 01_processed` がそれで済まない案件は、Codex が案件別スクリプトをその都度作る。

`normalize` は出力前に `responses_normalized.csv` の必須条件を自己検査し、重複 `response_id` や必須列空欄があれば失敗させる。
`screening` も出力前に `screened_responses.csv` を自己検査し、`screening_reason` と `is_target` の不整合を書き出さない。
`embeddings` も入力 `screened_responses.csv` と生成物の自己検査を行い、`completed` / `failed` の状態に合わない成果物を書き出さない。
`clustering` も入力 `screened_responses.csv` / `embeddings.npy` と生成物の自己検査を行い、`clusters.csv` と `clustering_metadata.json` の不整合を書き出さない。
`curation` も入力 `screened_responses.csv` / `clusters.csv` と生成物の自己検査を行い、不整合を書き出さない。
`classification` も入力 `screened_responses.csv` / `embeddings.npy` / `category_master.csv` と生成物を自己検査し、不整合を書き出さない。
標準フローの `classification.py` は単一ラベルのベクトル近傍分類を前提とする。

補助成果物:

- `99_logs/raw_to_processed_mapping.md`
- `03_embeddings/embedding_metadata.json`
- `03_embeddings/embedding_failures.csv` (失敗時のみ)
- `04_clustering/clustering_metadata.json`
- `05_curation/cluster_representatives.csv`
- `05_curation/category_master_draft.csv`
- `05_curation/curation_metadata.json`
- `06_classification/category_embeddings.npy`
- `06_classification/classification_metadata.json`

`embeddings` は同一入力・同一設定の既存成果物があれば再利用し、作り直したい場合だけ `--force` を付ける。
`clustering` も同一入力・同一設定の既存成果物があれば再利用し、作り直したい場合だけ `--force` を付ける。

CLI 一覧は `scripts/README.md` を参照。

詳細仕様は `docs/classification_pipeline_spec.md` を参照。
