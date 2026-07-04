# Scripts

公開I/F は `python scripts/pipeline.py ...` に固定する。  
個別スクリプトは内部実装として扱う。

## 基本

```bash
python scripts/pipeline.py init-project --project-name your_project_name
python scripts/pipeline.py init-question --project-name your_project_name --question-id Q1
```

`init-project` と `init-question` は template 内の sample 成果物を案件ディレクトリに複製しない。  
`init-question` は初期化済み project を前提にする。

## 実行

```bash
python scripts/pipeline.py normalize \
  --project-name your_project_name \
  --input projects/your_project_name/00_raw/source.csv \
  --response-id-col 回答ID \
  --question-id-col 設問ID \
  --question-text-col 質問文 \
  --answer-text-col 自由回答

python scripts/pipeline.py screening --project-name your_project_name
python scripts/pipeline.py embeddings --project-name your_project_name --question-id Q1
python scripts/pipeline.py clustering --project-name your_project_name --question-id Q1
python scripts/pipeline.py classification --project-name your_project_name --question-id Q1
python scripts/pipeline.py review --project-name your_project_name --question-id Q1
```

`pipeline.py` は `99_logs/pipeline.log` に共通実行ログを書き、工程ごとの `.log` も各出力先に追記する。  
各ステージは主成果物を書き出す前に自己検査し、不整合な成果物を残さない。

## 検査

```bash
python scripts/pipeline.py validate-processed --project-name your_project_name
python scripts/pipeline.py validate-screening --project-name your_project_name
python scripts/pipeline.py validate-question --project-name your_project_name --question-id Q1
python scripts/pipeline.py validate-project --project-name your_project_name
python scripts/pipeline.py validate-log --project-name your_project_name --log-name pipeline.log
```

`validate-question` は stage-aware で、未着手の後段成果物までは要求しない。  
`validate-project` も stage-aware で、未着手の後段成果物までは要求しない。
