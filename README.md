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

API 設定:

- OpenAI API を使うのは `embeddings.py`
- ルートの `.env` に `OPENAI_API_KEY` を置ける
- `.env.example` を `.env` にコピーして使う
- `.env` は Git 管理しない

```env
OPENAI_API_KEY=sk-...
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

`requirements.in` を更新したら、その後に `pip install -r requirements.txt` を実行する。

## ディレクトリ構成

プロジェクトごとの入力データと生成成果物は `projects/` 配下に分けて置く。

基本構成:

```txt
projects/
  {project_name}/
    00_raw/
    01_processed/
    02_screening/
    scripts/
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
- `05_curation` で代表回答を見て、人が `category_master.csv` と `topic_category_mapping.csv` を作る
- `05_curation` では `curation.py` が入力用の雛形も置くので、人はその続きを埋める
- `06_classification` で `topic_id` とカテゴリ対応を全件へ再適用する
- `03_embeddings` 以降は `questions/{question_id}/` 配下で設問ごとに進める

`projects/` 配下のプロジェクトデータと生成成果物は、原則 Git 管理しない。管理対象は `docs/`、コード、必要なら匿名化済みサンプルだけに絞る。

共通ドキュメントは `docs/`、今後の共通コードは `src/` または `scripts/` に分ける前提とする。
プロジェクト固有の raw 変換スクリプトは `projects/{project_name}/scripts/` に置き、共通の `scripts/` には入れない。

## 公開I/F

公開I/F は `python scripts/<script>.py ...` に固定する。

- プロジェクト作成は `init_project.py`
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
python scripts/curation.py --input projects/your_project_name/02_screening/screened_responses.csv --clusters projects/your_project_name/questions/Q1/04_clustering/clusters.csv --question-id Q1 --output-dir projects/your_project_name/questions/Q1/05_curation
python scripts/classification.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --clusters projects/your_project_name/questions/Q1/04_clustering/clusters.csv --category-master projects/your_project_name/questions/Q1/05_curation/category_master.csv --topic-category-mapping projects/your_project_name/questions/Q1/05_curation/topic_category_mapping.csv --output-dir projects/your_project_name/questions/Q1/06_classification
```

`clustering.py` の標準は `kmeans` / `k=100` にする。`hdbscan` を使う場合だけ `--clusterer hdbscan` を明示する。

`normalize.py` の標準機能は、1 CSV を標準4列へ写像する単純な列対応までとする。  
`00_raw -> 01_processed` がそれで済まないプロジェクトは、Codex がプロジェクト別スクリプトをその都度作る。
プロジェクト別スクリプトは `projects/{project_name}/scripts/` に置き、その project の raw にだけ使う。

`normalize` は出力前に `responses_normalized.csv` の必須条件を自己検査し、重複 `response_id` や必須列空欄があれば失敗させる。
`screening` も出力前に `screened_responses.csv` を自己検査し、`screening_reason` と `is_target` の不整合を書き出さない。
`embeddings` も入力 `screened_responses.csv` と生成物の自己検査を行い、`completed` / `failed` の状態に合わない成果物を書き出さない。
`clustering` も入力 `screened_responses.csv` / `embeddings.npy` と生成物の自己検査を行い、`clusters.csv` と `clustering_metadata.json` の不整合を書き出さない。
`curation` も入力 `screened_responses.csv` / `clusters.csv` と生成物の自己検査を行い、不整合な代表回答一覧を書き出さない。
`classification` も入力 `screened_responses.csv` / `clusters.csv` / `category_master.csv` / `topic_category_mapping.csv` と生成物を自己検査し、不整合を書き出さない。
標準フローの `classification.py` は `topic_id -> category_id` の単一ラベル再適用を前提とする。

補助成果物:

- `99_logs/raw_to_processed_mapping.md`
- `03_embeddings/embedding_metadata.json`
- `03_embeddings/embedding_failures.csv` (失敗時のみ)
- `04_clustering/clustering_metadata.json`
- `05_curation/cluster_representatives.csv`
- `05_curation/curation_metadata.json`
- `05_curation/topic_category_mapping.csv`
- `06_classification/classification_metadata.json`

`05_curation/category_master.csv` は `curation.py` が上書きしない。  
`05_curation/topic_category_mapping.csv` も `curation.py` が上書きしない。  
`curation.py` は未作成時だけ `category_master.csv` のヘッダと、通常 topic 一覧を入れた `topic_category_mapping.csv` の雛形を置く。  
人が `cluster_representatives.csv` を見てその 2 ファイルを追記・編集し、`classification.py` の入力として使う。

`embeddings` は同一入力・同一設定の既存成果物があれば再利用し、作り直したい場合だけ `--force` を付ける。
`clustering` も同一入力・同一設定の既存成果物があれば再利用し、作り直したい場合だけ `--force` を付ける。

## 運用の推奨

- embedding は `text-embedding-3-small` をデフォルトにしつつ、予算が許すなら `text-embedding-3-large` を優先して比較する
- clustering は `topic_id=-1` を減らす方向でパラメータを探る
- clustering の単一変更では、`hdbscan_min_samples` より `umap_n_neighbors` と `umap_n_components` の影響が大きいことが多い

この方針は標準フローを置き換えるものではなく、比較実験の優先順位を決めるための目安とする。

## 05_curation の進め方

`05_curation` は、人が BERTopic の山に業務上のカテゴリ名を付ける段階。

手順:

1. `python scripts/curation.py ...` を実行して `cluster_representatives.csv` を作る
2. `topic_id` ごとに代表回答と `topic_size` を見て、山の意味と大きさを確認する
3. 同じ意味の山は、同じ `category_id` に統合してよい
4. `category_master.csv` に正式なカテゴリ辞書を作る
5. `topic_category_mapping.csv` に `topic_id -> category_id` を記録する
6. その 2 ファイルを `classification.py` に渡して全件へ再適用する

`topic_category_mapping.csv` の記入ルール:

- 1 `topic_id` は 1 `category_id` にだけ対応させる
- 複数 `topic_id` を同じ `category_id` に統合してよい
- `topic_id=-1` は書かない
- 通常 topic は未対応のまま残さない

`category_master.csv` の記入ルール:

- `category_id` は一意にする
- `category_name` は集計・報告で使う正式名称にする
- `category_definition` は、そのカテゴリに含める意図が分かる短い説明にする

sample では、`topic_id=0` と `topic_id=2` を同じ `CAT001` に統合する例と、`topic_id=-1` が `OTHER` に落ちる例を含めている。

ローカルHTMLツールを使う場合:

1. `python scripts/curation.py ...` で `cluster_representatives.csv` を作る
2. `tools/curation_ui/index.html` をブラウザで開く
3. `cluster_representatives.csv` を読み込む
4. 必要なら既存の `topic_category_mapping.csv` と `category_master.csv` も読み込む
5. topic ごとの割当を編集し、`topic_category_mapping.csv` と `category_master.csv` をダウンロードする

このツールは `05_curation` の手作業を補助するUIで、`classification.py` の入力CSV仕様は変えない。

CLI 一覧は `scripts/README.md` を参照。

詳細仕様は `docs/README.md` を入口として参照。

## License

MIT
