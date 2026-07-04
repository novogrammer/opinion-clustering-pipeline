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
python scripts/pipeline.py duplicate-check --project-name your_project_name
python scripts/pipeline.py embeddings --project-name your_project_name --question-id Q1 --prepare-only
python scripts/pipeline.py embeddings --project-name your_project_name --question-id Q1
python scripts/pipeline.py clustering --project-name your_project_name --question-id Q1
python scripts/pipeline.py scaffold-category-master --project-name your_project_name --question-id Q1
python scripts/pipeline.py category-conflicts --project-name your_project_name --question-id Q1
python scripts/pipeline.py classification --project-name your_project_name --question-id Q1
python scripts/pipeline.py review --project-name your_project_name --question-id Q1
python scripts/pipeline.py review-summary --project-name your_project_name --question-id Q1
python scripts/pipeline.py review-priorities --project-name your_project_name --question-id Q1
python scripts/pipeline.py review-samples --project-name your_project_name --question-id Q1 --priority-trigger ambiguous_match
python scripts/pipeline.py review-corrections --project-name your_project_name --question-id Q1
python scripts/pipeline.py override-candidates --project-name your_project_name --question-id Q1
python scripts/pipeline.py promote-override-candidates --project-name your_project_name --question-id Q1
python scripts/pipeline.py override-rule-hits --project-name your_project_name --question-id Q1
python scripts/pipeline.py override-rule-summary --project-name your_project_name --question-id Q1
```

`pipeline.py` は `99_logs/pipeline.log` に共通実行ログを書き、工程ごとの `.log` も各出力先に自動で追記する。
`screening` は出力前に `screened_responses.csv` を自己検査し、`screening_reason` と `is_target` の不整合を書き出さない。
`duplicate-check` は出力前に `duplicate_responses.csv` を自己検査し、group 内件数や canonical 行の不整合を書き出さない。
`embeddings` は入力 `screened_responses.csv` と生成物の自己検査を行い、`prepared` / `completed` / `failed` の状態に合わない成果物を書き出さない。
`embeddings` は再試行上限を超えたレコードがあれば `03_embeddings/embedding_failures.csv` に隔離する。
`embeddings` は `--batch-size`, `--max-retries`, `--retry-base-seconds` を外部指定できる。
`embeddings` は一致する既存成果物があれば再利用し、強制再生成は `--force` を使う。
`clustering` は入力 `embedding_requests.csv` / `embeddings.npy` と生成物の自己検査を行い、`clusters.csv`、`cluster_summary.csv`、`clustering_metadata.json` の不整合を書き出さない。
`clustering` は UMAP/HDBSCAN の主要パラメータを外部指定できる。
`clustering` も一致する既存成果物があれば再利用し、強制再生成は `--force` を使う。
`classification` は入力 `screened_responses.csv`、`category_master.csv`、必要時の `manual_override_rules.csv` と生成物 `final_labels.csv` を自己検査し、不整合を書き出さない。
override 候補系は `review_corrections.csv`、`manual_override_candidates.csv`、`manual_override_rules.csv`、`override_rule_hits.csv`、`override_rule_summary.csv` の整合を自己検査し、不整合を書き出さない。
question 単位コマンドは、初期化済み `questions/{question_id}/` を前提にする。

## 検査

```bash
python scripts/pipeline.py validate-processed --project-name your_project_name
python scripts/pipeline.py validate-mapping --project-name your_project_name
python scripts/pipeline.py validate-screening --project-name your_project_name
python scripts/pipeline.py validate-duplicates --project-name your_project_name
python scripts/pipeline.py validate-embedding-requests --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-embedding-metadata --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-embedding-failures --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-embeddings-array --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-clusters --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-cluster-summary --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-clustering-metadata --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-category-master --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-override-rules --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-override-candidates --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-override-rule-hits --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-override-rule-summary --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-category-conflicts --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-final-labels --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-review-log --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-review-summary --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-review-priorities --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-review-samples --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-review-corrections --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-question --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-project --project-name your_project_name
python scripts/pipeline.py validate-log --project-name your_project_name --log-name pipeline.log
```

`validate-question` は stage-aware で、未着手の後段成果物までは要求しない。
`validate-project` も stage-aware で、未着手の後段成果物までは要求しない。

`classification.py` は `05_classification/manual_override_rules.csv` があれば自動で読み込む。  
定型語の例外分類や、レビューで見つかった個別ルールの差し込みに使う。

`review_samples.py` はレビュー対象の抜き出し用。  
既定では `high` を全件、`medium` をカテゴリごとに一部抽出し、必要なら trigger 指定で強制抽出できる。

`review_corrections.py` は人手修正が入った事例だけを抜き出す。  
override ルール追加やカテゴリ定義の見直し材料に使う。

`review_override_candidates.py` は修正事例から `manual_override_candidates.csv` を起こす。  
まずは安全側に `exact` マッチの下書きだけを出し、人が確認して `manual_override_rules.csv` に移す前提とする。

`promote_override_candidates.py` は `manual_override_candidates.csv` の `approved=true` 行だけを  
`manual_override_rules.csv` に反映する。既存ルールがあれば `rule_id` 単位で置換する。

`override_rule_hits.py` は `manual_override_rules.csv` がどの回答に当たるかを一覧化する。  
ルール追加後の効きすぎ・取りこぼし確認に使う。

`override_rule_summary.py` は `override_rule_hits.csv` を rule 単位に集約する。  
ヒット件数が多すぎる rule や、想定より広く当たる rule の確認に使う。

短縮呼び出しは `Makefile` を使う。
