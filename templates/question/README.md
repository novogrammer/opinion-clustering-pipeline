# Question Template

`init-question` が内部で使う `projects/{project_name}/questions/{question_id}/` のひな形。

- `03_embeddings/` の主成果物は `embeddings.npy`
- `04_clustering/` の主成果物は `clusters.csv`
- `05_curation/` の主成果物は `category_master.csv`
- `06_classification/` の主成果物は `final_labels.csv`
- `04_clustering/` の補助成果物は `clustering_metadata.json`
- `05_curation/` の補助成果物は `cluster_representatives.csv`, `category_master_draft.csv`, `curation_metadata.json`
- `06_classification/` の補助成果物は `category_embeddings.npy`, `classification_metadata.json`
- 各 stage の `.log` は監査用に残す
