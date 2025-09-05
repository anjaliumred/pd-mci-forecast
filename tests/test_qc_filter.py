import pytest
import pandas as pd
from scripts.qc_filter import norm_sub, norm_ses, find_confounds


def test_norm_sub():
    assert norm_sub("sub-001") == "sub-001"
    assert norm_sub("001") == "sub-001"
    assert norm_sub(123) == "sub-123"


def test_norm_ses():
    assert norm_ses("ses-01") == "ses-01"
    assert norm_ses("01") == "ses-01"
    assert norm_ses(2) == "ses-2"
    assert norm_ses("") is None
    assert norm_ses("none") is None
    assert norm_ses("null") is None
    assert norm_ses("nan") is None

# find_confounds requires a filesystem structure, so only basic call test

def test_find_confounds(tmp_path):
    deriv = tmp_path
    sub_id = "sub-001"
    ses_id = "ses-01"
    # Should return empty string if no files
    assert find_confounds(str(deriv), sub_id, ses_id) == ""
