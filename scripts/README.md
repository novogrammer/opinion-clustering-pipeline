# Scripts

公開I/F は `python scripts/<script>.py ...` に固定する。

## 基本

```bash
python scripts/init_project.py --project-name your_project_name
python scripts/init_question.py --project-dir projects/your_project_name --question-id Q1
```

`init_project.py` と `init_question.py` は template 内の sample 成果物をプロジェクトディレクトリに複製しない。  
`init_question.py` は初期化済み project を前提にする。

## 実行

```bash
python scripts/normalize.py \
  --input projects/your_project_name/00_raw/source.csv \
  --output projects/your_project_name/01_processed/responses_normalized.csv \
  --mapping-log projects/your_project_name/99_logs/raw_to_processed_mapping.md \
  --run-log projects/your_project_name/99_logs/raw_to_processed.log \
  --response-id-col 回答ID \
  --question-id-col 設問ID \
  --question-text-col 質問文 \
  --answer-text-col 自由回答

python scripts/screening.py --input projects/your_project_name/01_processed/responses_normalized.csv --output projects/your_project_name/02_screening/screened_responses.csv --log projects/your_project_name/99_logs/screening.log
python scripts/embeddings.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --output-dir projects/your_project_name/questions/Q1/03_embeddings
python scripts/clustering.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --embeddings projects/your_project_name/questions/Q1/03_embeddings/embeddings.npy --output-dir projects/your_project_name/questions/Q1/04_clustering
python scripts/curation.py --input projects/your_project_name/02_screening/screened_responses.csv --clusters projects/your_project_name/questions/Q1/04_clustering/clusters.csv --question-id Q1 --output-dir projects/your_project_name/questions/Q1/05_curation
python scripts/classification.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --clusters projects/your_project_name/questions/Q1/04_clustering/clusters.csv --category-master projects/your_project_name/questions/Q1/05_curation/category_master.csv --topic-category-mapping projects/your_project_name/questions/Q1/05_curation/topic_category_mapping.csv --output-dir projects/your_project_name/questions/Q1/06_classification
```

k-means 実験用に別経路を切る場合は、出力先だけ分けて `--clusterer kmeans` を付ける。

```bash
python scripts/clustering.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --embeddings projects/your_project_name/questions/Q1/03_embeddings/embeddings.npy --clusterer kmeans --output-dir projects/your_project_name/questions/Q1/04_clustering_kmeans
python scripts/curation.py --input projects/your_project_name/02_screening/screened_responses.csv --clusters projects/your_project_name/questions/Q1/04_clustering_kmeans/clusters.csv --question-id Q1 --output-dir projects/your_project_name/questions/Q1/05_curation_kmeans
python scripts/classification.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --clusters projects/your_project_name/questions/Q1/04_clustering_kmeans/clusters.csv --category-master projects/your_project_name/questions/Q1/05_curation_kmeans/category_master.csv --topic-category-mapping projects/your_project_name/questions/Q1/05_curation_kmeans/topic_category_mapping.csv --output-dir projects/your_project_name/questions/Q1/06_classification_kmeans
```

`clustering.py` は `clusters.csv` と `clustering_metadata.json` を出す。  
`curation.py` は `cluster_representatives.csv` を出す。  
未作成時だけ `topic_category_mapping.csv` の雛形と `category_master.csv` のヘッダも置く。  
`cluster_representatives.csv` では `topic_id` ごとの代表回答と `topic_size` を見る。  
人はその 1 CSV を見て `category_master.csv` と `topic_category_mapping.csv` を作成・編集し、`classification.py` へ渡す。  
ブラウザ内で編集したい場合は `tools/curation_ui/index.html` を開き、`cluster_representatives.csv` を読み込む。  
このUIは `topic_category_mapping.csv` と `category_master.csv` を再読込でき、標準仕様のままダウンロードする。  
`classification.py` は `final_labels.csv` と `classification_metadata.json` を出す。  
各ステージは主成果物を書き出す前に自己検査し、不整合な成果物を残さない。

## 最小確認

sample の対応関係:

- `templates/project/02_screening/screened_responses.sample.csv`
- `templates/question/04_clustering/clusters.sample.csv`
- `templates/question/05_curation/category_master.sample.csv`
- `templates/question/05_curation/topic_category_mapping.sample.csv`

この組み合わせで `classification.py` を実行すると、`templates/question/06_classification/final_labels.sample.csv` と同じ列構成の出力を確認できる。  
sample には、複数 topic の統合と `topic_id=-1` の outlier を含めている。  
`topic_category_mapping.sample.csv` を書き換えれば、未対応 topic の失敗も確認できる。
