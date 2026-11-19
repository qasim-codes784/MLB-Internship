import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW_FILE = next((BASE / f for f in ("student_performance.csv", "students.csv") if (BASE / f).exists()),
                BASE / "student_performance.csv")
CLEAN_FILE = BASE / "cleaned_student_performance.csv"
CHART_DIR = BASE / "charts"
SIGNATURE = "Written by: Syed M Qasim Bukhari"

NOT_A_SUBJECT = re.compile(r"id|roll|age|year|class|sem|average|total|percent|rank|grade|performance|attend", re.I)
NAME_HINT = re.compile(r"name", re.I)

def tidy(col):
    return re.sub(r"\s+", "_", str(col).strip()).title()

def detect_columns(df):
    name_col = next((c for c in df.columns if NAME_HINT.search(str(c))), None)
    if name_col is None:
        name_col = next((c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"), None)

    subjects = []
    for c in df.columns:
        if c == name_col or NOT_A_SUBJECT.search(str(c)):
            continue
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().mean() >= 0.5:
            subjects.append(c)
    return name_col, subjects

def find_subject(subjects, *keywords):
    for kw in keywords:
        for s in subjects:
            if kw.lower() in s.lower():
                return s
    return None

def load_clean():
    df = pd.read_csv(CLEAN_FILE)
    meta = {"Student_Name", "Average_Score", "Performance"}
    subjects = [c for c in df.columns
                if c not in meta and not NOT_A_SUBJECT.search(c) and pd.api.types.is_numeric_dtype(df[c])]
    return df, subjects
