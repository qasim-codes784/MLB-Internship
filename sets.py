data = [1, 2, 2, 3, 4, 4, 5, 5, 5]
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7}


def unique_values(items):
    # a set keeps only one copy of each value
    return set(items)


def union_of(a, b):
    # everything that is in a or b
    return a | b


def intersection_of(a, b):
    # only what is in both a and b
    return a & b


def difference_of(a, b):
    # what is in a but not in b
    return a - b


print("Original list:", data)
print("Unique values:", unique_values(data))
print("Unique (sorted list):", sorted(unique_values(data)))

print("Set A:", set_a)
print("Set B:", set_b)
print("Union:", union_of(set_a, set_b))
print("Intersection:", intersection_of(set_a, set_b))
print("Difference (A - B):", difference_of(set_a, set_b))
print("Difference (B - A):", difference_of(set_b, set_a))

print("-----------------------")
print("Written by: Syed M Qasim Bukhari")
