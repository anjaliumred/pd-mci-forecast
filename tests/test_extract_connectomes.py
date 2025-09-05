import pytest
import numpy as np
import pandas as pd
from scripts.extract_connectomes import norm_sub, norm_ses, first_hit, vec_upper, select_confounds

def test_norm_sub():
    assert norm_sub("sub-001") == "sub-001"
    assert norm_sub("001") == "sub-001"
    assert norm_sub(123) == "sub-123"

def test_norm_ses():
    assert norm_ses("ses-01") == "ses-01"
    assert norm_ses("01") == "ses-01"
    assert norm_ses("") is None
    assert norm_ses("none") is None
    assert norm_ses("null") is None
    assert norm_ses("nan") is None

def test_first_hit(tmp_path):
    # Should return empty string if no files
    assert first_hit([str(tmp_path / "nofile*.txt")]) == ""
    # Should return file if exists
    f = tmp_path / "file.txt"
    f.write_text("test")
    assert first_hit([str(f)]) == str(f)

def test_vec_upper():
    mat = np.array([[1,2,3],[2,4,5],[3,5,6]])
    v = vec_upper(mat)
    assert isinstance(v, np.ndarray)
    assert len(v) == 3

def test_select_confounds():
    df = pd.DataFrame({
        "trans_x": [0.1, 0.2],
        "trans_y": [0.1, 0.2],
        "rot_x": [0.1, 0.2],
        "white_matter": [0.1, 0.2],
        "a_comp_cor1": [0.1, 0.2],
        "a_comp_cor2": [0.1, 0.2],
    })
    arr = select_confounds(df, use_gsr=False, max_acompcor=2)
    assert arr.shape[1] == 6
