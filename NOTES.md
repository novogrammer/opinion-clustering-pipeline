# Notes

このファイルは、今回の削除対象から外した判断保留項目を残すためのメモ。

## 判断保留

- `scripts/validate_question_artifacts.py`
  - stage-aware な工程横断検査と cross-check が仕様に対して厳しすぎる可能性がある
  - ただし監査要件として妥当な範囲でもあり、今回は削除しない
- `scripts/validate_project_artifacts.py`
  - project 単位の横断検査は `validate-question` と同様に運用上は有用
  - 仕様の必須範囲として固定するかは未確定のため、今回は削除しない
- `templates/project/**.sample.*`
  - sample 成果物の同梱範囲が広い可能性がある
  - どこまでを正式テンプレートに含めるか未確定のため、今回は削除しない
- `templates/question/**.sample.*`
  - review / override / conflict 系の sample まで含めてよいかは未確定
  - 後続の仕様整理で再判断する
