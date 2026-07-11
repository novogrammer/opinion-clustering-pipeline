from __future__ import annotations

import pandas as pd

from classification import (
    build_final_labels_df,
    validate_cluster_alignment,
    validate_mapping_category_ids,
    validate_missing_non_outlier_topics,
)


def test_build_final_labels_merges_topics_and_maps_outlier_to_other() -> None:
    responses = pd.DataFrame(
        {
            "response_id": ["1", "2", "3", "4"],
            "question_id": ["Q1", "Q1", "Q1", "Q2"],
            "answer_text": ["安い", "低価格", "どれでもない", "別設問"],
            "is_target": ["true", "true", "true", "true"],
        }
    )
    clusters = pd.DataFrame(
        {
            "response_id": ["1", "2", "3"],
            "question_id": ["Q1", "Q1", "Q1"],
            "topic_id": ["0", "2", "-1"],
        }
    )
    mapping = pd.DataFrame(
        {
            "topic_id": ["0", "2"],
            "category_id": ["CAT001", "CAT001"],
        }
    )
    categories = pd.DataFrame(
        {
            "category_id": ["CAT001"],
            "category_name": ["価格"],
            "category_definition": ["価格に関する回答"],
        }
    )

    result = build_final_labels_df(
        responses_df=responses,
        clusters_df=clusters,
        mapping_df=mapping,
        category_master_df=categories,
        question_id="Q1",
    )

    assert result["response_id"].tolist() == ["1", "2", "3"]
    assert result["predicted_category_id"].tolist() == ["CAT001", "CAT001", "OTHER"]
    assert result["predicted_category_name"].tolist() == ["価格", "価格", "その他"]


def test_classification_input_validations_detect_mismatches() -> None:
    target_rows = pd.DataFrame(
        {"response_id": ["1", "2"], "question_id": ["Q1", "Q1"]}
    )
    clusters = pd.DataFrame(
        {
            "response_id": ["1", "3"],
            "question_id": ["Q1", "Q1"],
            "topic_id": ["0", "1"],
        }
    )
    mapping = pd.DataFrame({"topic_id": ["0"], "category_id": ["UNKNOWN"]})
    categories = pd.DataFrame({"category_id": ["CAT001"]})

    assert validate_cluster_alignment(target_rows, clusters) == [
        "clusters.csv responses do not match screened target responses"
    ]
    assert validate_mapping_category_ids(mapping, categories) == [
        "topic_category_mapping.csv contains category_id values not found in category_master.csv: UNKNOWN"
    ]
    assert validate_missing_non_outlier_topics(clusters, mapping) == [
        "topic_category_mapping.csv is missing topic_id values: 1"
    ]
