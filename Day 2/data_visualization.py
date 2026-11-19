import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import CHART_DIR, SIGNATURE, load_clean, find_subject

CHART_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
df, subjects = load_clean()

def save(name):
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{name}.png", dpi=150)
    plt.close()

order = df.sort_values("Average_Score", ascending=False)
plt.figure(figsize=(max(8, len(df) * 0.4), 5))
plt.bar(order["Student_Name"].astype(str), order["Average_Score"], color="steelblue")
plt.xticks(rotation=75, ha="right"); plt.ylabel("Average Score")
plt.title("Average Marks of Each Student")
save("1_bar_student_average")

plt.figure(figsize=(8, 5))
sns.histplot(df["Average_Score"], bins=10, kde=True, color="teal")
plt.xlabel("Average Score"); plt.title("Distribution of Average Scores")
save("2_hist_average_score")

x = find_subject(subjects, "python") or subjects[0]
y = find_subject(subjects, "machine", "ml") or subjects[1]
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x=x, y=y, hue="Performance", s=90)
plt.title(f"{x} vs {y} Marks (correlation = {df[x].corr(df[y]):.2f})")
save("3_scatter_subjects")

counts = df["Performance"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
        colors=sns.color_palette("pastel"))
plt.title("Students by Performance Category")
save("4_pie_performance")

plt.figure(figsize=(max(8, len(subjects) * 1.3), 5))
sns.boxplot(data=df[subjects])
plt.xticks(rotation=30); plt.ylabel("Marks"); plt.title("Spread of Marks by Subject")
save("5_box_subjects")

plt.figure(figsize=(8, 5))
plt.plot(subjects, df[subjects].mean().values, marker="o")
plt.xticks(rotation=30); plt.ylabel("Class Average"); plt.title("Class Average by Subject")
save("6_line_subject_average")

if "Course" in df.columns:
    plt.figure(figsize=(7, 5))
    by_course = df.groupby("Course")["Average_Score"].mean().sort_values(ascending=False)
    sns.barplot(x=by_course.index, y=by_course.values, hue=by_course.index, legend=False, palette="Set2")
    plt.ylabel("Mean Average Score"); plt.title("Average Score by Course")
    save("7_bar_course_average")

print(f"Charts saved in {CHART_DIR.name}/")
print("-----------------------\n" + SIGNATURE)
