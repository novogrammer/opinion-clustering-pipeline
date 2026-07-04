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
python scripts/classification.py --input projects/your_project_name/02_screening/screened_responses.csv --question-id Q1 --category-master projects/your_project_name/questions/Q1/05_classification/category_master.csv --output projects/your_project_name/questions/Q1/05_classification/final_labels.csv
python scripts/review.py --input projects/your_project_name/questions/Q1/05_classification/final_labels.csv --output projects/your_project_name/questions/Q1/06_review/review_log.csv --screened projects/your_project_name/02_screening/screened_responses.csv
```

各ステージは主成果物を書き出す前に自己検査し、不整合な成果物を残さない。

## 任意の横断検査

```bash
python scripts/validate_question.py --question-dir projects/your_project_name/questions/Q1
python scripts/validate_project.py --project-dir projects/your_project_name
python scripts/validate_log.py --input projects/your_project_name/99_logs/raw_to_processed.log
```

`validate_question.py` は stage-aware で、未着手の後段成果物までは要求しない。  
`validate_project.py` も stage-aware で、未着手の後段成果物までは要求しない。
