from __future__ import annotations

import numpy as np
import pandas as pd

from clustering import CLUSTER_COLUMNS, build_kmeans_clusters_df, build_single_topic_df


def test_empty_and_single_topic_fallbacks() -> None:
    empty_rows = pd.DataFrame(columns=["response_id", "question_id"])
    empty_embeddings = np.empty((0, 2), dtype=np.float32)

    empty_result, effective_k = build_kmeans_clusters_df(
        empty_rows,
        empty_embeddings,
        k=100,
        random_state=42,
    )

    assert effective_k == 0
    assert empty_result.empty
    assert empty_result.columns.tolist() == CLUSTER_COLUMNS

    single_rows = pd.DataFrame({"response_id": ["1"], "question_id": ["Q1"]})
    single_result = build_single_topic_df(single_rows)
    assert single_result.to_dict("records") == [
        {
            "response_id": "1",
            "question_id": "Q1",
            "topic_id": 0,
            "topic_probability": 1.0,
            "is_outlier": False,
        }
    ]


def test_kmeans_is_reproducible_and_caps_effective_k() -> None:
    target_rows = pd.DataFrame(
        {
            "response_id": ["1", "2", "3", "4"],
            "question_id": ["Q1"] * 4,
        }
    )
    embeddings = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        dtype=np.float32,
    )

    first, first_k = build_kmeans_clusters_df(
        target_rows,
        embeddings,
        k=100,
        random_state=42,
    )
    second, second_k = build_kmeans_clusters_df(
        target_rows,
        embeddings,
        k=100,
        random_state=42,
    )

    assert first_k == second_k == 4
    assert first.columns.tolist() == CLUSTER_COLUMNS
    assert first["topic_probability"].between(0.0, 1.0).all()
    pd.testing.assert_frame_equal(first, second)
