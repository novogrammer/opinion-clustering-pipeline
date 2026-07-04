# Question Template

`init-question` が内部で使う `projects/{project_name}/questions/{question_id}/` のひな形。

- `03_embeddings/` に `embeddings.npy`, `embedding_metadata.json`
- API 失敗時は `03_embeddings/embedding_failures.csv` に失敗レコードを残す
- `04_clustering/` に `clusters.csv`, `clustering_metadata.json`
- `05_classification/` に最終分類結果と `classification.log`
- `06_review/` にレビュー記録と `review.log`
- `03_embeddings/embedding.log`, `04_clustering/clustering.log` に工程ログを残す
