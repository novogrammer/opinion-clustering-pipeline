# Notes

- `04_clustering` の比較案として `k-means` を試す余地がある
  - 期待値: `topic_id=-1` の外れ値をなくせる
  - 前提: `clusters.csv` の列構成はできるだけ維持する
  - 保留論点: `k-means` では `topic_probability` の意味が変わるため、空欄許容にするか別指標へ置き換えるかを決める必要がある
