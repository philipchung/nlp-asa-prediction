def flatten_list_of_list(regular_list) -> list:
    "Flatten list of list, ignores any `None` items."
    output = []
    for sublist in (sublist for sublist in regular_list if sublist is not None):
        for item in (item for item in sublist if item is not None):
            output += [item]
    return output
