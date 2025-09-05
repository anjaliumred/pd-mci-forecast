import argparse
import os
import pandas as pd
from typing import Optional

def find_group_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the column in the DataFrame that likely contains group/diagnosis information.
    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        str or None: Name of group column or None if not found
    """
    cand = [c for c in df.columns if c.lower() in {"group","diagnosis","dx","cohort"} or "group" in c.lower() or "diag" in c.lower()]
    return cand[0] if cand else None

def main():
    """
    Main entry point for label creation script.
    Loads BIDS participants.tsv, finds group column, and writes labels CSV.
    """
    ap = argparse.ArgumentParser(description="Create labels CSV from BIDS participants.tsv.")
    ap.add_argument("--bids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = os.path.join(a.bids, "participants.tsv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, sep="\t")
    if "participant_id" in df.columns:
        df.rename(columns={"participant_id": "subject_id"}, inplace=True)
    gcol = find_group_column(df)
    if gcol is None:
        raise ValueError("Add a 'group' column to participants.tsv (HC/PD-NC/PD-MCI).")
    df.rename(columns={gcol: "group"}, inplace=True)
    out = df[["subject_id", "group"]].copy()
    out["session_id"] = ""
    out.to_csv(a.out, index=False)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()
