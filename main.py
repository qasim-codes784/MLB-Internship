students = []  # list of dictionaries, kept in memory only


def get_text(message):
    while True:
        value = input(message).strip()
        if value:
            return value
        print("This field cannot be empty.")


def get_name(message):
    while True:
        value = get_text(message)
        if value.replace(" ", "").isalpha():
            return value
        print("Name should contain letters only.")


def get_number(message, low, high):
    while True:
        value = input(message).strip()
        if value.isdigit() and low <= int(value) <= high:
            return int(value)
        print(f"Please enter a whole number between {low} and {high}.")


def find_student(roll_no):
    # return the student dictionary with this roll number, or None
    for student in students:
        if student["roll_no"] == roll_no:
            return student
    return None


def print_student(student):
    print(f"  Roll No: {student['roll_no']} | Name: {student['name']} | "
          f"Age: {student['age']} | Course: {student['course']}")


def add_student():
    print("\n--- Add Student ---")
    roll_no = get_number("Roll number: ", 1, 999999)
    if find_student(roll_no):
        print("A student with this roll number already exists.")
        return
    student = {
        "roll_no": roll_no,
        "name": get_name("Name: "),
        "age": get_number("Age (5-100): ", 5, 100),
        "course": get_text("Course: "),
    }
    students.append(student)
    print("Student added successfully.")


def view_students():
    print("\n--- All Students ---")
    if not students:
        print("No students found.")
        return
    for student in students:
        print_student(student)
    print("Total students:", len(students))


def search_student():
    print("\n--- Search Student ---")
    print("1. Search by roll number")
    print("2. Search by name")
    choice = input("Choose: ").strip()
    if choice == "1":
        student = find_student(get_number("Roll number: ", 1, 999999))
        if student:
            print_student(student)
        else:
            print("Student not found.")
    elif choice == "2":
        keyword = get_text("Name (or part of it): ").lower()
        found = [s for s in students if keyword in s["name"].lower()]
        if found:
            for student in found:
                print_student(student)
        else:
            print("Student not found.")
    else:
        print("Invalid choice.")


def update_student():
    print("\n--- Update Student ---")
    student = find_student(get_number("Roll number of student to update: ", 1, 999999))
    if not student:
        print("Student not found.")
        return
    print("Current record:")
    print_student(student)
    print("Press Enter to keep the current value.")

    name = input("New name: ").strip()
    if name:
        if name.replace(" ", "").isalpha():
            student["name"] = name
        else:
            print("Invalid name, keeping the old one.")

    age = input("New age: ").strip()
    if age:
        if age.isdigit() and 5 <= int(age) <= 100:
            student["age"] = int(age)
        else:
            print("Invalid age, keeping the old one.")

    course = input("New course: ").strip()
    if course:
        student["course"] = course

    print("Student updated successfully.")


def delete_student():
    print("\n--- Delete Student ---")
    student = find_student(get_number("Roll number of student to delete: ", 1, 999999))
    if not student:
        print("Student not found.")
        return
    students.remove(student)
    print("Student deleted successfully.")


def show_menu():
    print("\n===== Student Record Management System =====")
    print("1. Add Student")
    print("2. View All Students")
    print("3. Search Student")
    print("4. Update Student")
    print("5. Delete Student")
    print("6. Total Number of Students")
    print("7. Exit")


def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-7): ").strip()
        if choice == "1":
            add_student()
        elif choice == "2":
            view_students()
        elif choice == "3":
            search_student()
        elif choice == "4":
            update_student()
        elif choice == "5":
            delete_student()
        elif choice == "6":
            print("Total students:", len(students))
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please enter a number from 1 to 7.")

    print("-----------------------")
    print("Written by: Syed M Qasim Bukhari")


main()
