# Day 3 – Data Cleaning & Visualization

Dataset: `students.csv` (12 sample students, 3 subjects: Maths, Science, English).
> Note: the file is labelled sample/demo data, so the findings below describe this
> sample only, not real student results.

## How to run
```bash
pip install pandas matplotlib seaborn
python data_cleaning.py        # -> cleaned_student_performance.csv
python data_visualization.py   # -> 7 charts in charts/
python dashboard.py            # mini project: prints answers + charts/dashboard.png
```

## Data cleaning steps
1. Loaded the CSV, skipping the `#` comment lines at the top (`comment="#"`).
2. Checked missing values with `isnull().sum()`: 1 in `science`, 1 in `english`.
3. Checked duplicates: none found (`drop_duplicates()` is still applied).
4. Renamed columns (`roll_no` -> `Roll_No`, `name` -> `Student_Name`, `maths` -> `Maths`, etc.).
5. Converted marks to numeric with `pd.to_numeric`.
6. Filled the 2 missing marks with that subject's median.
7. Created `Average_Score` (mean of Maths, Science, English) and `Performance`
   (Excellent ≥ 90, Good 80–89, Average 70–79, Needs Improvement < 70).
8. Sorted by `Average_Score` (highest first) and saved `cleaned_student_performance.csv`.

## Visualizations and why
| Chart | File | Why |
|---|---|---|
| Bar – average per student | `1_bar_student_average.png` | Compare every student at a glance |
| Histogram | `2_hist_average_score.png` | See how average scores are spread |
| Scatter | `3_scatter_subjects.png` | Check if two subjects move together (the dataset has no Python/ML columns, so Maths vs Science is used) |
| Pie | `4_pie_performance.png` | Share of students per Performance category |
| Box plot | `5_box_subjects.png` | Spread, median and outliers per subject |
| Line | `6_line_subject_average.png` | Class average across subjects |
| Bar – by course | `7_bar_course_average.png` | Compare the three courses |
| Dashboard | `dashboard.png` | One-page summary for the mini project |

## Mini project answers
- Students: **12**
- Subject averages: English 70.25, Science 67.25, Maths 65.67
- Top 5: Sample Student H (94.33), C (86.33), A (84.33), K (82.00), F (74.33)
- Needs improvement: 7 students (J, E, G, B, I, D, L)
- Highest class average: **English**

## Key insights
1. **More than half the class needs improvement.** 7 of 12 students (58%) average below 70; only 1 is Excellent.
2. **Data Science students score lowest.** Their mean average is 59.3, versus about 73.7 for both AI and Computer Vision.
3. **Subjects rise and fall together.** Maths and Science marks are almost perfectly correlated (r ≈ 0.97), and Maths has the widest spread (std ≈ 21), so a student who is weak in one subject tends to be weak in all.

## Screen recording
_(add link / file here)_
