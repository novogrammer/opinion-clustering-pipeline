from __future__ import annotations

import pandas as pd

from screening import build_screened_dataframe, classify_answer, run_validations


def test_classify_answer_covers_standard_screening_reasons() -> None:
    assert classify_answer("") == (False, "blank")
    assert classify_answer(" 特になし ") == (False, "non_response")
    assert classify_answer("---") == (False, "symbol_only")
    assert classify_answer("料金を下げてほしい") == (True, "target")


def test_run_validations_detects_target_reason_mismatch() -> None:
    normalized = pd.DataFrame(
        {
            "response_id": ["1"],
            "question_id": ["Q1"],
            "question_text": ["改善点は？"],
            "answer_text": ["料金を下げてほしい"],
        }
    )
    screened = build_screened_dataframe(normalized)
    screened.loc[0, "is_target"] = False

    errors = run_validations(screened)

    assert "Row 1: is_target=false cannot have screening_reason=target" in errors
