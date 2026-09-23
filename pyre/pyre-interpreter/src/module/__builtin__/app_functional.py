"""
Plain Python definition of the builtin functions oriented towards
functional programming.
"""


def sorted(iterable, /, *, key=None, reverse=False):
    "sorted(iterable, key=None, reverse=False) --> new sorted list"
    sorted_lst = list(iterable)
    sorted_lst.sort(key=key, reverse=reverse)
    return sorted_lst
