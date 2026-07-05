# 05_curation UI

ローカルで `index.html` を開き、`cluster_representatives.csv` を読み込んで `topic_category_mapping.csv` と `category_master.csv` を作るための補助UI。

基本手順:

1. `python scripts/curation.py ...` を実行して `05_curation/cluster_representatives.csv` を作る
2. `tools/curation_ui/index.html` をブラウザで開く
3. `cluster_representatives.csv` を読み込む
4. 必要なら既存の `topic_category_mapping.csv` / `category_master.csv` も追加読込する
5. topic ごとに `category_id` を割り当てる
6. category 一覧で `category_name` と `category_definition` を整える
7. `topic_category_mapping.csv` と `category_master.csv` をダウンロードする
8. その 2 ファイルを `05_curation/` に置いて `classification.py` を実行する

ルール:

- `topic_id=-1` は outlier として表示するが、mapping には出力しない
- 通常 topic は未割当のままダウンロードできない
- 複数 topic を同じ `category_id` に統合してよい
- 出力CSVの列は標準仕様に合わせて固定する
