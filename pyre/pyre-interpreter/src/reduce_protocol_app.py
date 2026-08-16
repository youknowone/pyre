import sys

def reduce_1(obj, proto):
    import copyreg
    cls = obj.__class__

    # CPython 3.14 exposes object.__new__ on these native IO types, so
    # copyreg._reduce_ex walks through them to object and serializes a user
    # subclass via its __getstate__. PyPy's generic_new_descr (and pyre's
    # field-layout allocator) is necessarily a distinct callable, which makes
    # copyreg stop at the IO base and try to pickle a second native IO object.
    # Preserve the CPython-visible base decision only for a subclass; exact
    # Buffered*/TextIOWrapper objects remain deliberately unpickleable.
    io_object_new_bases = (
        '_io.BufferedReader',
        '_io.BufferedWriter',
        '_io.BufferedRandom',
        '_io.TextIOWrapper',
    )
    for base in cls.__mro__[1:]:
        if base is cls:
            continue
        if '%s.%s' % (base.__module__, base.__name__) in io_object_new_bases:
            getstate = obj.__getstate__
            state = getstate()
            # pyre cannot route object.__new__ through a distinct native IO
            # layout: PyPy's `check_user_subclass` correctly rejects that
            # allocator/layout pair.  `__newobj__` invokes the subtype's own
            # inherited native allocator and reaches the same CPython-visible
            # uninitialized instance before `__setstate__` restores it.
            args = (cls,)
            if state:
                return copyreg.__newobj__, args, state
            return copyreg.__newobj__, args
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
