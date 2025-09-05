import pytest
import pandas as pd
from scripts.make_labels import find_group_column

def test_find_group_column():
    df = pd.DataFrame({
        "participant_id": ["sub-001", "sub-002"],
        "diagnosis": ["A", "B"],
        "other": [1, 2]
    })
    assert find_group_column(df) == "diagnosis"
    df2 = pd.DataFrame({"participant_id": ["sub-001"], "other": [1]})
    assert find_group_column(df2) is None
