# docs

`docs/` は、このリポジトリの共通方針を用途別に整理した入口とする。

## 読む順番

1. [pipeline_spec.md](./pipeline_spec.md)
2. [review_criteria.md](./review_criteria.md)
3. [knowledge_notes.md](./knowledge_notes.md)

## 文書一覧

### [pipeline_spec.md](./pipeline_spec.md)

実装仕様の本体。

- パイプラインの対象範囲
- 入出力仕様
- 各 stage の主成果物
- 例外処理
- 再実行と監査の前提

「どう実装するか」を決める内容は、この文書を正本とする。

### [review_criteria.md](./review_criteria.md)

レビュー基準の本体。

- 良いクラスタ / 悪いクラスタの判断
- 粒度の判断
- `その他` の扱い
- 人手確認を優先する回答

「どう評価するか」を決める内容は、この文書を正本とする。

### [knowledge_notes.md](./knowledge_notes.md)

背景説明と知見の補足。

- Embedding と BERTopic の役割分担
- 採用モデルの考え方
- 実測メモ
- 実務フロー上の注意点

「なぜそうするか」「実測ではどうだったか」は、この文書を参照する。

## 更新ルール

- 実装仕様を変更する場合は `pipeline_spec.md` を更新する
- レビュー運用を変更する場合は `review_criteria.md` を更新する
- 背景説明、採用理由、実測値は `knowledge_notes.md` に寄せる
- 同じ内容を複数文書へ重複記載しない
- ルート `README.md` からは原則この文書を入口として案内する
