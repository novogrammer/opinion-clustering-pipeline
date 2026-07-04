# Question Template

`init-question` が内部で使う `projects/{project_name}/questions/{question_id}/` のひな形。

- `03_embeddings/` の主成果物は `embeddings.npy`
- `04_clustering/` の主成果物は `clusters.csv`
- `05_classification/` の主成果物は `final_labels.csv`
- 補助成果物は `embedding_metadata.json`, `embedding_failures.csv`, `clustering_metadata.json` に絞る
- 各 stage の `.log` は監査用に残す
