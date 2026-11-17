colors = ("red", "blue", "green", "blue", "red", "blue")
point = (3, 5, 7)


def count_occurrences(items, target):
   
    count = 0
    for item in items:
        if item == target:
            count += 1
    return count


def tuple_to_list(t):

    return list(t)


def list_to_tuple(lst):
  
    return tuple(lst)


print("Count of 'blue':", count_occurrences(colors, "blue"))
print("Count of 'red':", count_occurrences(colors, "red"))
print("Count using .count():", colors.count("blue"))

as_list = tuple_to_list(point)
print("Tuple to list:", as_list)
as_list.append(9)
print("List after append:", as_list)
print("List back to tuple:", list_to_tuple(as_list))

print("-----------------------")
print("Written by: Syed M Qasim Bukhari")
