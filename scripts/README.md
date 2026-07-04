# Scripts

公開I/F は `python scripts/pipeline.py ...` に固定する。  
個別スクリプトは内部実装として残す。

## 基本

```bash
python scripts/pipeline.py init-project --project-name your_project_name
python scripts/pipeline.py init-question --project-name your_project_name --question-id Q1
```

`init-project` と `init-question` は template 内の sample 成果物を案件ディレクトリに複製しない。
`init-question` は初期化済み project を前提にする。

## 01_processed

単純な1CSV正規化は共通 CLI を使う。

```bash
python scripts/pipeline.py normalize \
  --project-name your_project_name \
  --input projects/your_project_name/00_raw/source.csv \
  --response-id-col 回答ID \
  --question-id-col 設問ID \
  --question-text-col 質問文 \
  --answer-text-col 自由回答
```

`question_id` や `question_text` が列ではなく固定値なら `--question-id-value` や `--question-text-value` も使える。  
複数CSV結合や縦持ち変換が必要な案件は、Codex が案件別スクリプトを都度作る前提とし、これは公開I/Fに含めない。
`normalize` は出力前に `responses_normalized.csv` を自己検査し、重複 `response_id` や必須列空欄があれば停止する。
`normalize` を含む project 単位コマンドは、初期化済み `projects/{project_name}/` を前提にする。

## 実行

```bash
python scripts/pipeline.py screening --project-name your_project_name
python scripts/pipeline.py embeddings --project-name your_project_name --question-id Q1
python scripts/pipeline.py clustering --project-name your_project_name --question-id Q1
python scripts/pipeline.py classification --project-name your_project_name --question-id Q1
python scripts/pipeline.py review --project-name your_project_name --question-id Q1
```

`pipeline.py` は `99_logs/pipeline.log` に共通実行ログを書き、工程ごとの `.log` も各出力先に自動で追記する。
`screening` は出力前に `screened_responses.csv` を自己検査し、`screening_reason` と `is_target` の不整合や重複情報の不整合を書き出さない。
`embeddings` は入力 `screened_responses.csv` と生成物の自己検査を行い、`completed` / `failed` の状態に合わない成果物を書き出さない。
`embeddings` は再試行上限を超えたレコードがあれば `03_embeddings/embedding_failures.csv` に隔離する。
`embeddings` は `--batch-size`, `--max-retries`, `--retry-base-seconds` を外部指定できる。
`embeddings` は一致する既存成果物があれば再利用し、強制再生成は `--force` を使う。
`clustering` は入力 `screened_responses.csv` / `embeddings.npy` と生成物の自己検査を行い、`clusters.csv` と `clustering_metadata.json` の不整合を書き出さない。
`clustering` は UMAP/HDBSCAN の主要パラメータを外部指定できる。
`clustering` も一致する既存成果物があれば再利用し、強制再生成は `--force` を使う。
`classification` は入力 `screened_responses.csv` と `category_master.csv`、生成物 `final_labels.csv` を自己検査し、不整合を書き出さない。
question 単位コマンドは、初期化済み `questions/{question_id}/` を前提にする。

## 検査

```bash
python scripts/pipeline.py validate-processed --project-name your_project_name
python scripts/pipeline.py validate-mapping --project-name your_project_name
python scripts/pipeline.py validate-screening --project-name your_project_name
python scripts/pipeline.py validate-embedding-metadata --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-embedding-failures --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-embeddings-array --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-clusters --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-clustering-metadata --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-category-master --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-final-labels --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-review-log --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-question --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-project --project-name your_project_name
python scripts/pipeline.py validate-log --project-name your_project_name --log-name pipeline.log
```

`validate-question` は stage-aware で、未着手の後段成果物までは要求しない。
`validate-project` も stage-aware で、未着手の後段成果物までは要求しない。
