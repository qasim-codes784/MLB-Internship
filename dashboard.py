import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import CHART_DIR, SIGNATURE, load_clean

CHART_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
df, subjects = load_clean()

print("=" * 50, "\n STUDENT PERFORMANCE DASHBOARD\n", "=" * 50)

print(f"\n1. Total students: {len(df)}")

avg = df[subjects].mean().round(2).sort_values(ascending=False)
print("\n2. Average score per subject:\n" + avg.to_string())

top5 = df.nlargest(5, "Average_Score")[["Student_Name", "Average_Score"]]
print("\n3. Top 5 students:\n" + top5.to_string(index=False))

weak = df[df["Performance"] == "Needs Improvement"][["Student_Name", "Average_Score"]]
print("\n4. Students who need improvement:")
print(weak.to_string(index=False) if len(weak) else "   None - everyone is at 70 or above.")

print(f"\n5. Highest class average: {avg.index[0]} ({avg.iloc[0]})")

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Student Performance Dashboard", fontsize=16, fontweight="bold")

sns.barplot(x=avg.values, y=avg.index, ax=ax[0, 0], hue=avg.index, palette="viridis", legend=False)
ax[0, 0].set_title("Average Score per Subject"); ax[0, 0].set_xlabel("Average"); ax[0, 0].set_ylabel("")

sns.barplot(data=top5, x="Average_Score", y="Student_Name", ax=ax[0, 1],
            hue="Student_Name", palette="crest", legend=False)
ax[0, 1].set_title("Top 5 Students"); ax[0, 1].set_ylabel("")

counts = df["Performance"].value_counts()
ax[1, 0].pie(counts, labels=counts.index, autopct="%1.0f%%", startangle=90)
ax[1, 0].set_title("Performance Categories")

sns.boxplot(data=df[subjects], ax=ax[1, 1])
ax[1, 1].set_title("Spread of Marks"); ax[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(CHART_DIR / "dashboard.png", dpi=150)
print(f"\nDashboard saved -> {CHART_DIR.name}/dashboard.png")
print("-----------------------\n" + SIGNATURE)
