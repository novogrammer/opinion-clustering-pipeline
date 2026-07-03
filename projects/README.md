# Projects Directory

案件ごとの入力データと生成成果物は `projects/` 配下に分けて置く。

推奨構成:

```txt
projects/
  {project_name}/
    config/
      project.yaml
    data/
      raw/
      processed/
    embeddings/
    clustering/
    classification/
    review/
    logs/
```

例:

```txt
projects/customer_survey_2026q3/
projects/support_voice_nps_wave1/
```

`project_name` はディレクトリ名としてそのまま使う。半角英数字と `_` を基本にし、案件開始時に固定する。

共通ドキュメントは `docs/`、今後の共通コードは `src/` または `scripts/` に分ける前提とする。
