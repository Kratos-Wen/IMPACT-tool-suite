"""Lossless snapshots of mostly flat bbox rows without recursive scalar copying."""
from copy import deepcopy

_SCALARS = {str, int, float, bool, bytes, type(None)}


def clone_boxes(boxes):
    """Match deepcopy, including nested metadata, aliases and cyclic containers."""
    result = []
    memo = {id(boxes): result}
    for box in boxes:
        if type(box) is not dict:
            result.append(deepcopy(box, memo))
            continue
        if id(box) in memo:
            result.append(memo[id(box)])
            continue
        row = {}
        memo[id(box)] = row
        for key, value in box.items():
            row[key if type(key) in _SCALARS else deepcopy(key, memo)] = (
                value if type(value) in _SCALARS else deepcopy(value, memo)
            )
        result.append(row)
    return result
