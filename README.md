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
- `02_screening` で無回答や分類対象外を判定する
- `03_embeddings` 以降は `questions/{question_id}/` 配下で設問ごとに進める

`projects/` 配下の案件データと生成成果物は、原則 Git 管理しない。管理対象は `docs/`、コード、必要なら匿名化済みサンプルだけに絞る。

共通ドキュメントは `docs/`、今後の共通コードは `src/` または `scripts/` に分ける前提とする。

詳細仕様は `docs/classification_pipeline_spec.md` を参照。
