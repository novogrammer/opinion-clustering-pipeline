# 自由回答分類パイプライン実装仕様

## 目的

この文書は、[OpenAI APIを使った自由回答分類の整理メモ](/Users/novo/Documents/aptanastudio3workspace/opinion-clustering-pipeline/docs/openai_free_answer_classification_memo.md) を実装に落とすための仕様を定義する。

ここで扱うのは以下。

- 入力データの前提
- Codexによるデータ変形フロー
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

案件ごとに入力データ、生成Embedding、クラスタ結果、分類結果を分離する。

推奨構成:

```txt
docs/
projects/
  {project_name}/
    00_raw/
    01_processed/
    02_screening/
    questions/
      {question_id}/
        03_embeddings/
        04_clustering/
        05_classification/
    99_logs/
```

考え方:

- `docs/` は全案件共通の方針と仕様を置く
- `projects/{project_name}/` は案件固有データと成果物を置く
- 案件をまたいで再利用するコードは、今後 `src/` や `scripts/` に分離する
- 案件開始時のフォルダ作成はコマンドで都度行うのではなく、template から作る前提とする
- 段階をフォルダ名で明示し、案件の進行順が見えるようにする
- 連番を付けて並び順を固定する
- `00_raw` から `02_screening` までは案件全体で共有する
- `03_embeddings` 以降は `question_id` ごとに分ける

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

テンプレートは内部実装とし、利用者向けの公開I/Fは `python scripts/<script>.py ...` に固定する。  
各ステージは「1コマンドで実行し、主成果物を1つ置く」を原則とする。  
各ステージの自己検査は stage script の内部で完結させる。  
コマンド詳細は `README.md` と `scripts/README.md` に寄せ、この文書では工程仕様だけを扱う。

`init_project.py` と `init_question.py` は sample 成果物を実案件ディレクトリへ複製しない前提とする。
`init_question.py` は初期化済み project に対してだけ実行する前提とする。

---
---

## 入力データ仕様

1レコードは1回答とする。

必須カラム:

- `response_id`: 回答の一意ID
- `question_id`: 設問ID
- `question_text`: 質問文
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

## Codexによるデータ変形フロー

案件ごとの元データCSVは、そのまま本パイプライン入力に合うとは限らない。  
必要なデータ形式への変形は、案件開始時に Codex に依頼する運用を前提とする。

基本フロー:

1. 元データを `projects/{project_name}/00_raw/` に置く
2. `python scripts/normalize.py ...` で済むならその公開I/Fを使う
3. それで足りない案件だけ、Codex に案件別の変換処理を作らせる
4. 変換後データを `projects/{project_name}/01_processed/` に保存する
5. その後の分類対象判定は `projects/{project_name}/02_screening/` で行う
6. `question_id` ごとに `questions/{question_id}/` 配下で Embedding、クラスタリング、分類を進める

公開 CLI の前提:

- project 作成は `scripts/init_project.py`
- question 作成は `scripts/init_question.py`
- `normalize.py` の標準機能は、1 CSV を標準4列へ写像する単純な列対応まで
- question 単位処理は `scripts/embeddings.py` 以降を直接呼ぶ

Codex に依頼する内容の例:

- CSV から必要列だけを抽出する
- 列名を `response_id`, `question_id`, `question_text`, `answer_text` に合わせる
- 複数CSVファイルを結合する
- アンケート固有コード値を人間可読な値に直す
- 設問マスタと回答データを結合する
- 1行に複数回答が入っている形式を縦持ちへ変換する

運用ルール:

- 元データは `00_raw/` に保持し、直接上書きしない
- Codex が行った変形は、再実行できる形で残す
- 変形内容は案件ごとに `99_logs/` に記録する
- 変換後データのカラム仕様はこの文書の入力データ仕様に合わせる

---

## バージョン管理方針

`projects/` 配下の案件データと生成成果物は、原則として Git 管理しない。

原則として Git 管理しない対象:

- `00_raw/`
- `01_processed/`
- `02_screening/`
- `questions/{question_id}/` 配下の生成成果物

理由:

- 生データはサイズが大きくなりやすい
- 個人情報や機微情報を含む可能性がある
- 生成成果物は再作成できるため、Git差分の価値が低い

Git 管理するもの:

- `docs/`
- コード
- 必要なら匿名化済みの小さなサンプルデータ

版の追跡方法:

- 元データのファイル名に受領日や版を入れる
- `99_logs/` に使用した元データ名と変換内容を残す

---

## 判定前の整形ルール

`00_raw -> 01_processed` では判断を入れず、形式だけをそろえる。

必須ルール:

- 前後空白を除去する
- 改行、タブ、連続空白は正規化する
- Unicode正規化方針を固定する
- URL、メールアドレス、電話番号など個人情報になりうる文字列はマスク対象にできるようにする
- 1行1回答に整形する
- 列名を標準化する

推奨ルール:

- 表記揺れ辞書は最初から入れすぎない
- 日本語の誤字脱字補正は原則しない
- 回答文の要約や言い換えは行わない

この段階では以下を行わない。

```txt
無回答判定
分類対象外判定
短文だが有効な回答かどうかの判断
```

`01_processed/responses_normalized.csv` の完成条件:

- CSVである
- UTF-8で保存されている
- 1行1回答になっている
- ヘッダが `response_id`, `question_id`, `question_text`, `answer_text` で固定されている
- 列名が標準化されている
- 改行、タブ、連続空白が正規化されている
- 空欄や `なし` などの値も、この段階では削除せず残っている

共通 `normalize` CLI を使う場合は、この完成条件を出力前に自己検査し、条件を満たさない結果を書き出さない。

---

## 分類対象判定

`01_processed -> 02_screening` で、分類対象に入れるかどうかを判定する。

初期運用では、`02_screening` は Codex や共通スクリプトから実行する自動判定を前提とする。  
ここで行うのは、文字列ルールで決まる単純なスクリーニングに限る。

この段階で扱うもの:

- 無回答判定
- 実質無回答判定
- 分類対象外レコードの切り分け

無回答候補の例:

```txt
""
"なし"
"特になし"
"-"
"n/a"
```

ルール:

- 判定は `01_processed` には書き戻さない
- 判定結果は `02_screening` に保存する
- クラスタリングと分類には、分類対象レコードだけを流す
- 無回答や対象外レコードも件数管理のため残す
- 初期ルールは文字列一致や記号判定で決まるものに絞る
- 文脈解釈が必要な判定は、この段階では扱わない
- 最初は単純なルールで始め、必要が出たら `02_screening` のスクリプトを更新していく
- 無回答候補の辞書は、初期運用では別ファイルに分けず `02_screening` のスクリプト内に持つ
- `02_screening` の共通スクリプトは `scripts/` 配下に置く
- 入力CSVと出力CSVのパスは CLI 引数で渡す
- 標準の受け渡しCSV名は `responses_normalized.csv` と `screened_responses.csv` に固定する

初期判定フロー:

1. `answer_text` の前後空白を除去する
2. 空文字なら `is_target=false`, `screening_reason=blank`
3. 正規化後の文字列が無回答辞書に完全一致したら `is_target=false`, `screening_reason=non_response`
4. 記号だけで構成されていたら `is_target=false`, `screening_reason=symbol_only`
5. それ以外は `is_target=true`, `screening_reason=target`

この段階で無理にやらないこと:

- 短文だから対象外とみなす判定
- 設問文との意味関係を見て有効回答か判断すること
- 皮肉、婉曲表現、文脈依存の実質無回答を判定すること

`non_response` の初期候補例:

```txt
なし
特になし
とくになし
ない
わからない
なしです
n/a
na
no answer
```

`symbol_only` の例:

```txt
-
--
---
?
??
...
/
```

`screened_responses.csv` の想定カラム:

- `response_id`
- `question_id`
- `question_text`
- `answer_text`
- `is_target`
- `screening_reason`

考え方:

- 最初の4列は `01_processed` をそのまま引き継ぐ
- `is_target` は分類対象なら `true`、対象外なら `false`
- `screening_reason` は対象外理由や判定理由を記録する
- 初期運用では `screened_responses.csv` の1ファイルで管理する

初期運用で使う `screening_reason`:

```txt
blank
non_response
symbol_only
target
```

初期方針:

- `blank`: 空文字
- `non_response`: `なし`、`特になし`、`n/a` などの実質無回答
- `symbol_only`: 記号のみ
- `target`: 分類対象

この段階では `too_short` は使わない。  
短文が有効回答かどうかは判断を要するため、初期ルールから外す。

共通 `screening` CLI を使う場合は、この判定語彙と `is_target` の整合を出力前に自己検査し、不整合な `screened_responses.csv` を書き出さない。

---

## Embedding生成仕様

初期採用モデル:

```txt
text-embedding-3-small
```

Embedding入力は回答本文のみではなく、設問文脈を含めた1文字列とする。

入力元:

- `projects/{project_name}/02_screening/screened_responses.csv`
- `is_target = true` のレコードだけを使う
- `question_id` ごとに `projects/{project_name}/questions/{question_id}/03_embeddings/` を作る

テンプレート:

```txt
設問ID: {question_id}
質問: {question_text}
回答: {answer_text}
```

ルール:

- 同一分析単位ではEmbeddingモデルを混在させない
- 同一設問に対しては同じテンプレートを使う
- `02_screening` で分類対象と判定されたレコードだけをEmbedding対象とする
- `embedding_input_text` はテンプレートから機械的に生成する
- `03_embeddings` の主成果物は `embeddings.npy` とする

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

- `projects/{project_name}/01_processed/responses_normalized.csv`
- `projects/{project_name}/02_screening/screened_responses.csv`
- `projects/{project_name}/questions/{question_id}/03_embeddings/embedding_failures.csv` (失敗時のみ)
- `projects/{project_name}/99_logs/raw_to_processed_mapping.md`
- `projects/{project_name}/questions/{question_id}/03_embeddings/embeddings.npy`
- `projects/{project_name}/questions/{question_id}/03_embeddings/embedding_metadata.json`
- `projects/{project_name}/questions/{question_id}/04_clustering/clusters.csv`
- `projects/{project_name}/questions/{question_id}/05_classification/final_labels.csv`

`embedding_metadata.json` に含める項目:

- model
- question_id
- row_count
- input_screened_path
- input_screened_sha1
- input_template_version
- input_template
- preprocessing_version
- batch_size
- failed_count
- status
- created_at

共通 `embeddings` CLI を使う場合は、入力 `screened_responses.csv`、`embedding_metadata.json`、必要に応じて `embeddings.npy` / `embedding_failures.csv` の整合を出力前に自己検査し、`completed` / `failed` の状態に合わない成果物を書き出さない。

保存方針:

- 同一入力で同一設定なら再生成しない
- 成果物名に日付を埋め込むより、メタデータで管理する
- 上書き時は実行ログに差分が残るようにする

---

## クラスタリング仕様

クラスタリングは `question_id` 単位で実行する。

理由:

- 設問が異なる回答を混在させない
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

`clustering_metadata.json` に含める項目:

- row_count
- input_screened_path
- input_screened_sha1
- input_embeddings_path
- input_embeddings_sha1
- parameters
- created_at

共通 `clustering` CLI を使う場合は、入力 `screened_responses.csv` / `embeddings.npy` と、出力 `clusters.csv`、`clustering_metadata.json` の整合を出力前に自己検査し、不整合な成果物を書き出さない。

標準補助成果物:

- `cluster_representatives.csv`
- `cluster_label_drafts.csv`

`cluster_representatives.csv` の想定カラム:

- `topic_id`
- `response_id`
- `question_id`
- `answer_text`
- `topic_probability`
- `representative_rank`

`cluster_label_drafts.csv` の想定カラム:

- `topic_id`
- `draft_category_name`
- `draft_category_definition`
- `draft_representative_examples`
- `draft_confidence`
- `split_suggestion`

`cluster_label_drafts.csv` は LLM による草案であり、最終分類の決定権は持たない。  
人は `cluster_representatives.csv` と `cluster_label_drafts.csv` を見て `category_master.csv` を整える。

注意点:

- 外れ値トピックは通常カテゴリと分けて扱う
- 小規模設問ではクラスタリングより直接分類のほうが適切な場合がある
- パラメータ変更時の結果比較は、標準成果物ではなく任意の補助作業として扱う

---

## 最終分類仕様

人手でカテゴリ定義を確定した後、全件分類を行う。

入力:

- `question_id`
- `screened_responses.csv`
- `embeddings.npy`
- `category_master.csv`

`category_master` に必要な項目:

- `category_id`
- `category_name`
- `category_definition`
- `representative_examples`

出力:

- `response_id`
- `question_id`
- `answer_text`
- `predicted_category_id`
- `predicted_category_name`
- `confidence`
- `reason`
- `needs_human_review`

補助成果物:

- `category_embeddings.npy`
- `classification_metadata.json`

共通 `classification` CLI を使う場合は、入力 `screened_responses.csv` / `embeddings.npy` / `category_master.csv` と、出力 `final_labels.csv`、`category_embeddings.npy`、`classification_metadata.json` の整合を出力前に自己検査し、不整合な成果物を書き出さない。

標準フローの `classification` は単一ラベル分類を前提とする。  
複数ラベル分類は標準フローには含めず、必要になった時点で別途仕様化する。

分類方針:

- 回答側は `03_embeddings` の既存 embedding を再利用する
- カテゴリ側は `category_master.csv` の 1 行を 1 ベクトルとして embedding 化する
- カテゴリ embedding 用テキストは `category_name`, `category_definition`, `representative_examples` を連結して作る
- cosine similarity で最も近いカテゴリを採用する
- 標準フローでは単一ラベルで分類する
- confidence が閾値未満のときは `OTHER` と人手確認に回す

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

- `question_id` ごとのログ
- API再試行ログ
- 元データからパイプライン入力への変換ログ

保存先例:

- `projects/{project_name}/99_logs/raw_to_processed.log`
- `projects/{project_name}/99_logs/screening.log`
- `projects/{project_name}/questions/{question_id}/03_embeddings/embedding.log`
- `projects/{project_name}/questions/{question_id}/04_clustering/clustering.log`
- `projects/{project_name}/questions/{question_id}/05_classification/classification.log`

ログ形式:

- ログファイルは JSONL を前提とする
- 各行は少なくとも `event` と `created_at` を持つ

ログ検査:

```bash
python scripts/validate_log.py \
  --input projects/{project_name}/99_logs/raw_to_processed.log
```

再実行方針:

- `01_processed` が同じなら `02_screening` を再実行できる
- `02_screening` の対象レコードが同じならEmbeddingを再利用する
- クラスタリング条件だけ変えた場合はEmbeddingを再利用する
- カテゴリ定義変更時は最終分類のみ再実行可能にする

---

## 最初の実装範囲

最初の1本目では、以下までできれば十分。

1. CSVを読み込む
2. 1行1回答・4列の `01_processed` を作る
3. `02_screening` で分類対象判定を行う
4. `question_id` ごとにEmbeddingを生成して保存する
5. `question_id` ごとにクラスタリングする
6. `final_labels.csv` まで出せる
