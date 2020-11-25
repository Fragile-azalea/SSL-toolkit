def unroll(target):
    if isinstance(target, list) or isinstance(target, tuple):
        return sum((unroll(idx) for idx in target), [])
    else:
        return [target, ]
