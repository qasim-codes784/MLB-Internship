# Day 1 – Python Fundamentals, Data Structures & GitHub Basics

**Name:** Syed M Qasim Bukhari
**GitHub:** [qasim-codes784](https://github.com/qasim-codes784)
**Repository:** https://github.com/qasim-codes784/MLB-Internship

## What I Learned Today
- Virtual environments (`venv`): what they are and how to create and activate one.
- Variables, data types, operators, and conditional statements.
- Functions with `def`, parameters, and `return`.
- Lists, tuples, sets, and dictionaries, and when to use each.
- Git and GitHub basics: repository, commit, push, clone, branching.

## Project Structure
```text
Day1/
├── README.md
├── Practice_Problems/
│   ├── lists.py
│   ├── tuples.py
│   ├── sets.py
│   └── dictionaries.py
├── Student_Grading_System/
│   └── main.py
└── Student_Record_Management/
    └── main.py
```

## Practice Problems
- **Lists:** largest, second largest, remove duplicates, reverse without `reverse()`, common elements.
- **Tuples:** count occurrences, convert tuple to list and back.
- **Sets:** unique values, union and intersection.
- **Dictionaries:** student record, average marks, word frequency.

## Mini Projects
1. **Student Grading System:** takes name, class, subjects and marks, calculates the average, and assigns a grade (A, B, C, F).
2. **Student Record Management System (v1):** console menu to add, view, search, update, and delete students, stored in memory using a list of dictionaries.

## How to Run
```powershell
cd MLB-Internship\Day1
python Practice_Problems\lists.py
python Student_Grading_System\main.py
python Student_Record_Management\main.py
```

## Challenges Faced
- Mixing up Command Prompt and PowerShell commands (`New-Item` only works in PowerShell).
- Creating the project in the wrong folder and having to move it into `MLB-Internship`.
- The virtual environment broke when the folder moved, so I recreated it.
- Understanding how to save files with the correct `.py` extension and run them from the right folder.

## Solutions Implemented
- Used PowerShell for PowerShell commands and checked my location with `pwd`.
- Moved the folder with `Move-Item` and recreated the venv with `python -m venv venv`.
- Added a `.gitignore` containing `venv/` so the environment is not uploaded.
- Added the lists, tuples, sets, and dictionaries solutions and tested each one by running it.

## Author
Syed M Qasim Bukahri 