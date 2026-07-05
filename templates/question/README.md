# Question Template

`init-question` が内部で使う `projects/{project_name}/questions/{question_id}/` のひな形。

- `03_embeddings/` の主成果物は `embeddings.npy`
- `04_clustering/` の主成果物は `clusters.csv`
- `05_curation/` の主成果物は `cluster_representatives.csv`, `category_master.csv`, `topic_category_mapping.csv`
- `06_classification/` の主成果物は `final_labels.csv`
- `04_clustering/` の補助成果物は `clustering_metadata.json`
- `05_curation/` の補助成果物は `curation_metadata.json`
- `06_classification/` の補助成果物は `classification_metadata.json`
- `cluster_representatives.csv` には各 `topic_id` の代表回答と `topic_size` が入る
- sample には、複数 topic を同じ category に統合する例と、`topic_id=-1` の outlier 例を含めている
- `category_master.csv` は人が作成・編集する確定版で、`curation.py` は上書きしない
- `topic_category_mapping.csv` は人が作成・編集する確定版で、`curation.py` は上書きしない
- 各 stage の `.log` は監査用に残す
