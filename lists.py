numbers = [4, 9, 2, 9, 7, 1]
other = [7, 3, 9, 5]


def largest(items):
    
    biggest = items[0]
    for item in items:
        if item > biggest:
            biggest = item
    return biggest


def second_largest(items):
   
    first = None
    second = None
    for item in items:
        if first is None or item > first:
            second = first
            first = item
        elif item != first and (second is None or item > second):
            second = item
    return second


def remove_duplicates(items):
  
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def reverse_list(items):
   
    result = []
    for i in range(len(items) - 1, -1, -1):
        result.append(items[i])
    return result


def common_elements(a, b):
  
    result = []
    for item in a:
        if item in b and item not in result:
            result.append(item)
    return result


print("Largest:", largest(numbers))
print("Second largest:", second_largest(numbers))
print("No duplicates:", remove_duplicates(numbers))
print("Reversed:", reverse_list(numbers))
print("Common:", common_elements(numbers, other))

print("-----------------------")
print("Written by: Syed M Qasim Bukhari")
