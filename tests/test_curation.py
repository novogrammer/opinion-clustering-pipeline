from __future__ import annotations

import pandas as pd

from curation import build_representatives_df, build_topic_category_mapping_template_df


def test_build_representatives_orders_and_limits_each_topic() -> None:
    target_rows = pd.DataFrame(
        {
            "response_id": ["2", "1", "3", "4", "5"],
            "question_id": ["Q1"] * 5,
            "answer_text": ["two", "one", "three", "four", "five"],
        }
    )
    clusters = pd.DataFrame(
        {
            "response_id": ["2", "1", "3", "4", "5"],
            "question_id": ["Q1"] * 5,
            "topic_id": ["0", "0", "0", "0", "1"],
            "topic_probability": ["0.90", "0.90", "0.50", "0.70", "0.80"],
            "is_outlier": ["false"] * 5,
        }
    )

    result = build_representatives_df(clusters, target_rows)

    topic_zero = result[result["topic_id"] == "0"]
    assert topic_zero["response_id"].tolist() == ["1", "2", "4"]
    assert topic_zero["representative_rank"].tolist() == [1, 2, 3]
    assert topic_zero["topic_size"].tolist() == [4, 4, 4]
    assert result[result["topic_id"] == "1"]["response_id"].tolist() == ["5"]


def test_mapping_template_excludes_outlier_deduplicates_and_sorts_numerically() -> None:
    clusters = pd.DataFrame({"topic_id": ["10", "-1", "2", "2"]})

    result = build_topic_category_mapping_template_df(clusters)

    assert result.to_dict("records") == [
        {"topic_id": "2", "category_id": ""},
        {"topic_id": "10", "category_id": ""},
    ]
