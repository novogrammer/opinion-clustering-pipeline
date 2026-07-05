# Scripts

公開I/F は `python scripts/<script>.py ...` に固定する。

## 基本

```bash
python scripts/init_project.py --project-name your_project_name
python scripts/init_question.py --project-dir projects/your_project_name --question-id Q1
```

`init_project.py` と `init_question.py` は template 内の sample 成果物を案件ディレクトリに複製しない。  
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

`clustering.py` は `clusters.csv` と `clustering_metadata.json` を出す。  
`curation.py` は `cluster_representatives.csv` を出す。  
人は `cluster_representatives.csv` を見て `category_master.csv` と `topic_category_mapping.csv` を作成・編集し、`classification.py` へ渡す。  
`classification.py` は `final_labels.csv` と `classification_metadata.json` を出す。  
各ステージは主成果物を書き出す前に自己検査し、不整合な成果物を残さない。
