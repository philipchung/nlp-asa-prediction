from __future__ import annotations

import time
from functools import wraps


def ranges(nums: list) -> list[tuple]:
    """Given list of integers, returns list with consecutive integers as ranges.
    If list has single number range `x`, will return `(x, x)`.
    For Input: [2, 3, 4, 7, 8, 9, 15]
    Output looks like: [(2, 4), (7, 9), (15, 15)]
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def find_intersecting_sets(list_of_sets: list[set]) -> list[set]:
    """Given list of sets, find sets that intersect and merge them with
    a union operation, then returns the list of sets.  If set does not
    intersect with others, then it is returned without modification.
    """
    output_list = []
    while len(list_of_sets) > 0:
        # Pop sets of indices one by one
        setA = list_of_sets.pop()
        # If first set, add to output list
        if not output_list:
            output_list.append(setA)
        # For later sets, check if set overlaps with existing output_list items
        else:
            intersected = False
            for setB in output_list:
                intersect = setA.intersection(setB)
                # If overlaps, merge them and replace grp in output_list
                if bool(intersect):
                    output = setA.union(setB)
                    output_list.remove(setB)
                    output_list.append(output)
                    intersected = True
            if not intersected:
                output_list.append(setA)
    return output_list


def timer(f):
    "Decorator used for timing functions."

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"`{f.__name__}` took: {te-ts:2.4f} sec")
        return result

    return wrap


def timing(f):
    "Decorator used for timing functions and also displaying arguments"

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} args:[{args, kw}] took: {te-ts:2.4f} sec")
        return result

    return wrap
    