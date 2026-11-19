import pandas as pd
from common import RAW_FILE, CLEAN_FILE, SIGNATURE, detect_columns, tidy

if not RAW_FILE.exists():
    raise SystemExit(f"Put students.csv next to this script ({RAW_FILE.parent}).")

df = pd.read_csv(RAW_FILE, comment="#")
print("Shape:", df.shape)
print("\n--- Data types (before) ---\n", df.dtypes)

print("\n--- Missing values per column ---\n", df.isnull().sum())

print("\nDuplicate rows found:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

name_col, subjects = detect_columns(df)
df.columns = [tidy(c) for c in df.columns]
subjects = [tidy(s) for s in subjects]
if name_col is not None:
    df = df.rename(columns={tidy(name_col): "Student_Name"})
else:
    df.insert(0, "Student_Name", [f"Student {i+1}" for i in range(len(df))])
print("\nSubjects detected:", subjects)

for col in subjects:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Student_Name"])
df = df.dropna(subset=subjects, how="all")
for col in subjects:
    df[col] = df[col].fillna(df[col].median())

df["Average_Score"] = df[subjects].mean(axis=1).round(2)

def performance(score):
    if score >= 90: return "Excellent"
    if score >= 80: return "Good"
    if score >= 70: return "Average"
    return "Needs Improvement"

df["Performance"] = df["Average_Score"].apply(performance)

df = df.sort_values("Average_Score", ascending=False).reset_index(drop=True)
df.to_csv(CLEAN_FILE, index=False)

print("\nMissing values left:", int(df.isnull().sum().sum()))
print(df.head(), "\n\nSaved ->", CLEAN_FILE.name)
print("-----------------------\n" + SIGNATURE)
