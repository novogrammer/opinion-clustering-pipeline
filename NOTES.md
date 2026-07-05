# Notes

- `04_clustering` の比較案として `k-means` を試す余地がある
  - 期待値: `topic_id=-1` の外れ値をなくせる
  - 前提: `clusters.csv` の列構成はできるだけ維持する
  - 保留論点: `k-means` では `topic_probability` の意味が変わるため、空欄許容にするか別指標へ置き換えるかを決める必要がある
- embedding model は `text-embedding-3-small` だけでなく `text-embedding-3-large` も比較検討する
  - 観点: `topic_id=-1` 比率、topic 数、代表回答の解釈しやすさ、人手でカテゴリ付けしやすいか
  - 前提: 次元数の違いは現行実装で吸収できる
  - 保留論点: 標準を `small` のままにするか、`large` を比較用の標準外実験に留めるかは未決
