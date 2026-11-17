student = {
    "name": "Ali",
    "roll_no": 101,
    "age": 20,
    "course": "Computer Science",
    "marks": {"Math": 85, "English": 78, "Science": 92},
}


def average_marks(record):
    # add up all marks and divide by how many subjects there are
    total = 0
    for mark in record["marks"].values():
        total += mark
    return total / len(record["marks"])


def word_frequency(sentence):
    # count how many times each word appears
    counts = {}
    for word in sentence.lower().split():
        word = word.strip(".,!?;:")
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


print("Student record:", student)
print("Name:", student["name"])
print("Course:", student.get("course"))

student["age"] = 21  # update a value
student["email"] = "ali@example.com"  # add a new key
print("After update:", student)

print("Average marks: %.2f" % average_marks(student))

text = "the cat and the dog and the bird"
print("Sentence:", text)
print("Word frequency:", word_frequency(text))

print("-----------------------")
print("Written by: Syed M Qasim Bukhari")
