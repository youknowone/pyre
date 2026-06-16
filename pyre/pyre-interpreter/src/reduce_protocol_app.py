import sys

def reduce_1(obj, proto):
    import copyreg
    return copyreg._reduce_ex(obj, proto)

def reduce_2(obj, proto, args, kwargs):
    cls = obj.__class__

    # Py_TPFLAGS_DISALLOW_INSTANTIATION (1 << 7): a type whose tp_new is
    # NULL (generator / frame / ...) cannot be reconstructed via
    # __newobj__, so reduce_newobj refuses it before building the tuple.
    if type(obj).__flags__ & (1 << 7):
        raise TypeError("cannot pickle %r object" % type(obj).__name__)

    try:
        copyreg = sys.modules['copyreg']
    except KeyError:
        import copyreg

    if not isinstance(args, tuple):
        raise TypeError("__getnewargs__ should return a tuple")
    if not kwargs:
        newobj = copyreg.__newobj__
        args2 = (cls,) + args
    else:
        newobj = copyreg.__newobj_ex__
        args2 = (cls, args, kwargs)
    state = obj.__getstate__()
    listitems = iter(obj) if isinstance(obj, list) else None
    dictitems = iter(obj.items()) if isinstance(obj, dict) else None

    return newobj, args2, state, listitems, dictitems


def get_slotvalues(obj):
    names = slotnames(obj.__class__)
    if not names:
        return None
    slots = {}
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            pass
        else:
            slots[name] = value
    return slots


def slotnames(cls):
    if not isinstance(cls, type):
        return None

    try:
        return cls.__dict__["__slotnames__"]
    except KeyError:
        pass

    try:
        copyreg = sys.modules['copyreg']
    except KeyError:
        import copyreg
    slotnames = copyreg._slotnames(cls)
    if not isinstance(slotnames, list) and slotnames is not None:
        raise TypeError("copyreg._slotnames didn't return a list or None")
    return slotnames
