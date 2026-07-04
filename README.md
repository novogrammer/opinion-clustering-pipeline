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
        05_classification/
        06_review/
    99_logs/
```

- `01_processed` は判断なしの整形結果
- `02_screening` で無回答や分類対象外を文字列ルールで自動判定する
- `02_screening` では文脈解釈をせず、空欄、定型無回答、記号のみを機械的に切り分ける
- `02_screening` では重複回答の監査用抽出もできる
- `05_classification` はカテゴリマスタを使ったルールベース分類の土台まで実装済み
- `03_embeddings` 以降は `questions/{question_id}/` 配下で設問ごとに進める

`projects/` 配下の案件データと生成成果物は、原則 Git 管理しない。管理対象は `docs/`、コード、必要なら匿名化済みサンプルだけに絞る。

共通ドキュメントは `docs/`、今後の共通コードは `src/` または `scripts/` に分ける前提とする。

## 公開I/F

公開I/F は `python scripts/pipeline.py <command> ...` だけに固定する。

- 案件作成は `init-project`
- 設問作成は `init-question`
- 生CSVの標準化は `normalize --input ...`
- 以降の工程は `screening` から `review` までを順に実行する
- 個別スクリプト直実行と `templates/` の直接操作は公開I/Fに含めない
- `init-project` と `init-question` は sample 成果物を複製しない
- `init-question` 以外の project 単位コマンドは、初期化済み `projects/{project_name}/` を前提にする
- question 単位コマンドは、初期化済み `questions/{question_id}/` を前提にする

標準フロー:

```bash
python scripts/pipeline.py init-project --project-name your_project_name
python scripts/pipeline.py init-question --project-name your_project_name --question-id Q1
python scripts/pipeline.py normalize --project-name your_project_name --input projects/your_project_name/00_raw/source.csv --response-id-col 回答ID --question-id-col 設問ID --question-text-col 質問文 --answer-text-col 自由回答
python scripts/pipeline.py screening --project-name your_project_name
python scripts/pipeline.py duplicate-check --project-name your_project_name
python scripts/pipeline.py embeddings --project-name your_project_name --question-id Q1 --prepare-only
python scripts/pipeline.py clustering --project-name your_project_name --question-id Q1
python scripts/pipeline.py scaffold-category-master --project-name your_project_name --question-id Q1
python scripts/pipeline.py classification --project-name your_project_name --question-id Q1
python scripts/pipeline.py review --project-name your_project_name --question-id Q1
```

`00_raw -> 01_processed` が単純な列対応で済まない案件は、Codex が案件別スクリプトをその都度作る。  
それは内部作業であり、公開I/Fには含めない。

`normalize` は出力前に `responses_normalized.csv` の必須条件を自己検査し、重複 `response_id` や必須列空欄があれば失敗させる。
`screening` も出力前に `screened_responses.csv` を自己検査し、`screening_reason` と `is_target` の不整合を書き出さない。
`duplicate-check` も出力前に `duplicate_responses.csv` を自己検査し、group 内件数や canonical 行の不整合を書き出さない。
`embeddings` も入力 `screened_responses.csv` と生成物の自己検査を行い、`prepared` / `completed` / `failed` の状態に合わない成果物を書き出さない。
`clustering` も入力 `embedding_requests.csv` / `embeddings.npy` と生成物の自己検査を行い、`clusters.csv`、`cluster_summary.csv`、`clustering_metadata.json` の不整合を書き出さない。
`classification` も入力 `screened_responses.csv`、`category_master.csv`、必要時の `manual_override_rules.csv` と生成物 `final_labels.csv` を自己検査し、不整合を書き出さない。
override 候補系も `review_corrections.csv`、`manual_override_candidates.csv`、`manual_override_rules.csv`、`override_rule_hits.csv`、`override_rule_summary.csv` の整合を自己検査し、不整合を書き出さない。

補助成果物:

- `05_classification/`
  - `category_conflicts.csv`
  - `manual_override_candidates.csv`
  - `manual_override_rules.csv`
  - `override_rule_hits.csv`
  - `override_rule_summary.csv`
- `06_review/`
  - `review_summary.csv`
  - `category_review_priorities.csv`
  - `review_samples.csv`
  - `review_corrections.csv`

検査例:

```bash
python scripts/pipeline.py validate-processed --project-name your_project_name
python scripts/pipeline.py validate-screening --project-name your_project_name
python scripts/pipeline.py validate-question --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-project --project-name your_project_name
```

`validate-question` は進んでいる段階までを前提に検査する。  
たとえば `category_master.csv` だけを作った段階では、まだ `final_labels.csv` や `review_log.csv` を必須にしない。

`validate-project` も同様に stage-aware で、`init-project` 直後の未着手状態では後段成果物を要求しない。

`embeddings` は同一入力・同一設定の既存成果物があれば再利用し、作り直したい場合だけ `--force` を付ける。
`clustering` も同一入力・同一設定の既存成果物があれば再利用し、作り直したい場合だけ `--force` を付ける。

CLI 一覧は `scripts/README.md`、短縮呼び出しは `Makefile` を参照。

詳細仕様は `docs/classification_pipeline_spec.md` を参照。
