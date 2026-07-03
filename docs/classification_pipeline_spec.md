# 自由回答分類パイプライン実装仕様

## 目的

この文書は、[OpenAI APIを使った自由回答分類の整理メモ](/Users/novo/Documents/aptanastudio3workspace/opinion-clustering-pipeline/docs/openai_free_answer_classification_memo.md) を実装に落とすための仕様を定義する。

ここで扱うのは以下。

- 入力データの前提
- 前処理ルール
- Embedding生成方法
- クラスタリング実行単位
- 中間成果物の保存方法
- 最終分類の入出力仕様
- 例外処理
- 再実行性とログ方針

---

## スコープ

本パイプラインは、自由回答のカテゴリ発見と全件分類を対象にする。

- フェーズ1: Embedding + クラスタリングによる分類候補の発見
- フェーズ2: 人手で確定したカテゴリを用いた全件分類

以下はスコープ外とする。

- ダッシュボード実装
- レポート文章の自動生成
- 個別案件ごとの業務定義そのものの決定

---

## 推奨フォルダ構造

案件ごとに入力データ、生成Embedding、クラスタ結果、レビュー結果を分離する。

推奨構成:

```txt
docs/
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

考え方:

- `docs/` は全案件共通の方針と仕様を置く
- `projects/{project_name}/` は案件固有データと成果物を置く
- 案件をまたいで再利用するコードは、今後 `src/` や `scripts/` に分離する

`project_name` の例:

```txt
customer_survey_2026q3
app_store_reviews_japan_2026_07
support_voice_nps_wave1
```

命名ルール:

- 半角英数字と `_` を使う
- 日付や wave 番号など、後で識別に必要な情報を含める
- 表記ゆれを防ぐため案件開始時に固定する
- ディレクトリ名としてそのまま使う

---

## 入力データ仕様

1レコードは1回答とする。

必須カラム:

- `response_id`: 回答の一意ID
- `question_id`: 設問ID
- `question_text`: 質問文
- `question_intent`: 設問意図。なければ空文字可
- `answer_text`: 自由回答本文

推奨カラム:

- `survey_id`: 調査ID
- `respondent_id`: 回答者ID。匿名化済みの内部IDのみ
- `submitted_at`: 回答日時
- `segment_*`: 属性セグメント。性別、年代、契約種別など

制約:

- UTF-8で扱えること
- 改行を含んでもよい
- `response_id` は重複不可
- `question_id` は同一設問内で安定していること

---

## 前処理ルール

前処理では、意味を壊さずに比較可能性を上げることを優先する。

必須ルール:

- 前後空白を除去する
- 空文字、記号のみ、`-`、`なし` などの実質無回答を判定する
- 改行、タブ、連続空白は正規化する
- Unicode正規化方針を固定する
- URL、メールアドレス、電話番号など個人情報になりうる文字列はマスク対象にできるようにする

推奨ルール:

- 表記揺れ辞書は最初から入れすぎない
- 日本語の誤字脱字補正は原則しない
- 回答文の要約や言い換えは行わない

無回答判定例:

```txt
""
"なし"
"特になし"
"-"
"n/a"
```

無回答はクラスタリング対象外として別管理する。

---

## Embedding生成仕様

初期採用モデル:

```txt
text-embedding-3-small
```

Embedding入力は回答本文のみではなく、設問文脈を含めた1文字列とする。

テンプレート:

```txt
設問ID: {question_id}
設問意図: {question_intent}
質問: {question_text}
回答: {answer_text}
```

ルール:

- 同一分析単位ではEmbeddingモデルを混在させない
- 同一設問に対しては同じテンプレートを使う
- 前処理後文字列をEmbedding対象とする
- `question_intent` が空なら空欄のままテンプレートに入れる

バッチ処理方針:

- API投入は複数件まとめて行う
- バッチサイズはトークン量を見て調整する
- 失敗時はバッチ全体を再送するのではなく、再試行単位を切り分けられる設計にする

再現性のため、以下を必ず記録する。

- 使用モデル名
- テンプレート文字列
- 前処理バージョン
- 実行日時

---

## 保存成果物

中間成果物は再利用可能な形で保存する。

最低限必要な成果物:

- `projects/{project_name}/data/processed/responses_normalized.csv`
- `projects/{project_name}/data/processed/embedding_inputs.csv`
- `projects/{project_name}/embeddings/embeddings_{question_id}.npy`
- `projects/{project_name}/embeddings/embedding_metadata_{question_id}.json`
- `projects/{project_name}/clustering/clusters_{question_id}.csv`
- `projects/{project_name}/clustering/cluster_summary_{question_id}.csv`
- `projects/{project_name}/classification/final_labels_{question_id}.csv`
- `projects/{project_name}/review/review_log.csv`

`embedding_metadata_{question_id}.json` に含める項目:

- model
- question_id
- row_count
- input_template_version
- preprocessing_version
- created_at

保存方針:

- 同一入力で同一設定なら再生成しない
- 成果物名に日付を埋め込むより、メタデータで管理する
- 上書き時は実行ログに差分が残るようにする

---

## クラスタリング仕様

クラスタリングは `question_id` 単位で実行する。

理由:

- 設問意図が異なる回答を混在させない
- カテゴリ粒度を設問ごとに制御しやすい

初期方針:

- BERTopic を使用する
- Embeddingは外部で生成したものを渡す
- UMAP + HDBSCAN の標準構成を前提とする

実装上、最低限パラメータとして外に出すもの:

- `umap_n_neighbors`
- `umap_n_components`
- `hdbscan_min_cluster_size`
- `hdbscan_min_samples`
- `random_state`

出力カラム例:

- `response_id`
- `question_id`
- `topic_id`
- `topic_probability`
- `is_outlier`

注意点:

- 外れ値トピックは通常カテゴリと分けて扱う
- 小規模設問ではクラスタリングより直接分類のほうが適切な場合がある
- パラメータ変更時は結果比較用のサマリを残す

---

## クラスタ要約仕様

各クラスタについて、人手判断しやすい要約テーブルを作る。

最低限含める項目:

- `question_id`
- `topic_id`
- `cluster_size`
- `representative_answers`
- `candidate_label`
- `candidate_definition`
- `include_criteria`
- `exclude_criteria`
- `split_suggestion`
- `confidence`

代表回答抽出方針:

- クラスタ中心に近い回答だけに寄せない
- 境界に近い回答も含める
- 少数派の表現も確認できるようにする
- 同文の重複例はなるべく省く

---

## 最終分類仕様

人手でカテゴリ定義を確定した後、全件分類を行う。

入力:

- `question_id`
- `question_text`
- `question_intent`
- `answer_text`
- `category_master`

`category_master` に必要な項目:

- `category_id`
- `category_name`
- `category_definition`
- `include_criteria`
- `exclude_criteria`
- `example_positive`
- `example_negative`

出力:

- `response_id`
- `question_id`
- `predicted_category_id`
- `predicted_category_name`
- `confidence`
- `reason`
- `needs_human_review`

分類方針:

- 単一ラベルか複数ラベルかを設問ごとに決める
- どのカテゴリにも当てはまらない場合のラベルを定義する
- 判断材料不足のときに保留できるようにする

---

## 人手確認フロー

以下は人手確認対象に回す。

- `confidence` が閾値未満
- `その他` に分類された回答
- 複数カテゴリ候補が競合した回答
- 回答が短すぎて意味が不安定なもの
- 個人情報や攻撃的表現を含むもの

確認後は、修正結果を再学習用データではなく評価ログとして保持する。

---

## 例外処理

最低限扱うべき例外:

- 空回答
- 極端に短い回答
- 長文で複数論点が混ざる回答
- 同一回答の重複
- API失敗
- レート制限
- モデル変更

実装ルール:

- API失敗時は指数バックオフ付きで再試行する
- 永続失敗したレコードはID付きで隔離する
- モデル変更時は過去のEmbeddingを再利用しない
- 入力テンプレート変更時もEmbedding再生成対象にする

---

## ログ・監査・再実行

再現可能性のため、各実行で以下を残す。

- 実行日時
- 入力ファイルのパスまたはバージョン
- 使用モデル
- 使用パラメータ
- 対象件数
- 成功件数
- 失敗件数
- 出力先

推奨ログ単位:

- ジョブ全体ログ
- `question_id` ごとのログ
- API再試行ログ

保存先例:

- `projects/{project_name}/logs/pipeline.log`
- `projects/{project_name}/logs/embedding_{question_id}.log`
- `projects/{project_name}/logs/clustering_{question_id}.log`

再実行方針:

- 正規化済み入力が同じならEmbeddingを再利用する
- クラスタリング条件だけ変えた場合はEmbeddingを再利用する
- カテゴリ定義変更時は最終分類のみ再実行可能にする

---

## 最初の実装範囲

最初の1本目では、以下までできれば十分。

1. CSVを読み込む
2. 無回答を除外して正規化する
3. `question_id` ごとにEmbeddingを生成して保存する
4. `question_id` ごとにクラスタリングする
5. クラスタ要約CSVを出す

全件への最終分類は、その後の第2段階とする。
