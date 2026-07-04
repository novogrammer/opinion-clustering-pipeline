# Question Template

`init-question` が内部で使う `projects/{project_name}/questions/{question_id}/` のひな形。

- `03_embeddings/` の主成果物は `embeddings.npy`
- `04_clustering/` の主成果物は `clusters.csv`
- `05_classification/` の主成果物は `final_labels.csv`
- `04_clustering/` の補助成果物は `clustering_metadata.json`, `cluster_representatives.csv`, `cluster_label_drafts.csv`
- `05_classification/` の補助成果物は `category_embeddings.npy`, `classification_metadata.json`
- 各 stage の `.log` は監査用に残す
