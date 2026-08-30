import _ast
import platform
import sys
import types
import warnings

from testutils import assert_raises

ns = types.SimpleNamespace(a=2, b="Rust")

assert ns.a == 2
assert ns.b == "Rust"
with assert_raises(AttributeError):
    _ = ns.c


def _module_type_receiver_contract():
    for name, args, message in (
        (
            "__repr__",
            ({},),
            "descriptor '__repr__' requires a 'module' object but received a 'dict'",
        ),
        (
            "__getattribute__",
            ({}, "name"),
            "descriptor '__getattribute__' requires a 'module' object but received a 'dict'",
        ),
        (
            "__dir__",
            ({},),
            "descriptor '__dir__' for 'module' objects doesn't apply to a 'dict' object",
        ),
        (
            "__init__",
            ({}, "name"),
            "descriptor '__init__' requires a 'module' object but received a 'dict'",
        ),
    ):
        with assert_raises(TypeError) as raised:
            getattr(types.ModuleType, name)(*args)
        assert str(raised.exception) == message


_module_type_receiver_contract()


def _singleton_type_receiver_contract():
    for owner, name, message in (
        (
            types.EllipsisType,
            "__repr__",
            "descriptor '__repr__' requires a 'ellipsis' object but received a 'dict'",
        ),
        (
            types.EllipsisType,
            "__reduce__",
            "descriptor '__reduce__' for 'ellipsis' objects doesn't apply to a 'dict' object",
        ),
        (
            types.NotImplementedType,
            "__bool__",
            "descriptor '__bool__' requires a 'NotImplementedType' object but received a 'dict'",
        ),
        (
            types.NoneType,
            "__hash__",
            "descriptor '__hash__' requires a 'NoneType' object but received a 'dict'",
        ),
    ):
        with assert_raises(TypeError) as raised:
            getattr(owner, name)({})
        assert str(raised.exception) == message


_singleton_type_receiver_contract()


def _generator_type_contract():
    def empty_generator():
        yield None

    def local_generator(a, b, c):
        x = a + b
        yield x + c

    def closure_factory(value):
        def closure_generator():
            yield value

        return closure_generator

    def cell_generator(value):
        def read_cell():
            return value

        yield read_cell

    shared_order = [
        "__repr__",
        "__next__",
        "send",
        "throw",
        "close",
        "__iter__",
        "gi_running",
        "gi_suspended",
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "__name__",
        "__qualname__",
        "__doc__",
    ]
    if sys.implementation.name == "pyre":
        assert [
            name for name in types.GeneratorType.__dict__ if name in shared_order
        ] == shared_order
    if hasattr(types, "ClassMethodDescriptorType"):
        assert type(types.GeneratorType.__repr__) is types.WrapperDescriptorType
        assert type(types.GeneratorType.send) is types.MethodDescriptorType
        assert (
            type(types.GeneratorType.__dict__["__class_getitem__"])
            is types.ClassMethodDescriptorType
        )

    for function, args in (
        (empty_generator, ()),
        (local_generator, (1, 2, 3)),
        (closure_factory(1), ()),
        (cell_generator, (1,)),
    ):
        generator = function(*args)
        if hasattr(generator, "__sizeof__"):
            code = generator.gi_code
            nlocalsplus = (
                code.co_nlocals
                + sum(name not in code.co_varnames for name in code.co_cellvars)
                + len(code.co_freevars)
            )
            expected = types.GeneratorType.__basicsize__ + (
                nlocalsplus + code.co_stacksize
            ) * tuple.__itemsize__
            assert generator.__sizeof__() == expected
        generator.close()

    if sys.implementation.name != "pypy":
        with assert_raises(TypeError) as raised:
            types.GeneratorType.send({}, None)
        assert str(raised.exception) == (
            "descriptor 'send' for 'generator' objects doesn't apply to a 'dict' object"
        )


_generator_type_contract()


def _coroutine_type_contract():
    async def coroutine(value):
        return value

    shared_order = [
        "__repr__",
        "send",
        "throw",
        "close",
        "__await__",
        "cr_running",
        "cr_suspended",
        "cr_frame",
        "cr_code",
        "cr_await",
        "cr_origin",
        "__name__",
        "__qualname__",
        "__doc__",
    ]
    if sys.implementation.name == "pyre":
        assert [
            name for name in types.CoroutineType.__dict__ if name in shared_order
        ] == shared_order
    if hasattr(types, "ClassMethodDescriptorType"):
        assert type(types.CoroutineType.__repr__) is types.WrapperDescriptorType
        assert type(types.CoroutineType.send) is types.MethodDescriptorType
        assert (
            type(types.CoroutineType.__dict__["__class_getitem__"])
            is types.ClassMethodDescriptorType
        )

    instance = coroutine(1)
    if hasattr(instance, "__sizeof__"):
        code = instance.cr_code
        nlocalsplus = (
            code.co_nlocals
            + sum(name not in code.co_varnames for name in code.co_cellvars)
            + len(code.co_freevars)
        )
        expected = types.CoroutineType.__basicsize__ + (
            nlocalsplus + code.co_stacksize
        ) * tuple.__itemsize__
        assert instance.__sizeof__() == expected
    instance.close()

    unawaited = coroutine(2)
    frame = unawaited.cr_frame
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert unawaited.__del__() is None
    assert unawaited.cr_frame is frame
    assert len(recorded) == 1
    assert recorded[0].category is RuntimeWarning
    assert str(recorded[0].message) == (
        "coroutine '_coroutine_type_contract.<locals>.coroutine' was never awaited"
    )
    unawaited.close()

    if sys.implementation.name != "pypy":
        with assert_raises(TypeError) as raised:
            types.CoroutineType.send({}, None)
        assert str(raised.exception) == (
            "descriptor 'send' for 'coroutine' objects doesn't apply to a 'dict' object"
        )


_coroutine_type_contract()


def _function_type_kwdefaults():
    def source(a, /, b, *, c):
        return a + b + c

    function = types.FunctionType(
        source.__code__, {}, "function", (1, 2), None, {"c": 3}
    )
    assert function() == 6
    assert function.__defaults__ == (1, 2)
    assert function.__kwdefaults__ == {"c": 3}

    with assert_raises(TypeError) as raised:
        types.FunctionType(source.__code__, {}, "function", None, None, 3)
    assert str(raised.exception) == "arg 6 (kwdefaults) must be None or dict"


_function_type_kwdefaults()


def _function_type_closure_contract():
    def plain():
        return None

    def outer():
        value = 42

        def inner():
            return value

        return inner

    closed = outer()
    cell = closed.__closure__[0]

    # Both upstream constructors inspect argdefs before closure.  CPython 3.14
    # additionally requires a tuple here, while PyPy's `fixedview` accepts the
    # list and continues on to reject the closure.
    with assert_raises(TypeError) as raised:
        types.FunctionType(plain.__code__, {}, None, 42, 42)
    assert str(raised.exception) == "arg 4 (defaults) must be None or tuple"

    with assert_raises(TypeError) as raised:
        types.FunctionType(plain.__code__, {}, None, [], object())
    assert str(raised.exception) == "arg 4 (defaults) must be None or tuple"

    with assert_raises(TypeError) as raised:
        types.FunctionType(closed.__code__, globals())
    assert str(raised.exception) == "arg 5 (closure) must be tuple"

    with assert_raises(ValueError) as raised:
        types.FunctionType(closed.__code__, globals(), "inner", None, ())
    assert str(raised.exception) == "inner requires closure of length 1, not 0"

    with assert_raises(TypeError) as raised:
        types.FunctionType(closed.__code__, globals(), "inner", None, (42,))
    assert str(raised.exception) == "arg 5 (closure) expected cell, found int"

    class TupleSubclass(tuple):
        pass

    closure = TupleSubclass((cell,))
    clone = types.FunctionType(closed.__code__, globals(), "inner", None, closure)
    assert clone() == 42
    assert clone.__closure__ is closure

    empty_closure = ()
    without_closure = types.FunctionType(
        plain.__code__, globals(), "plain", None, empty_closure
    )
    assert without_closure.__closure__ is empty_closure


_function_type_closure_contract()


def _union_unhashable_partition_is_stable():
    is_hashable = False

    class UnhashableMeta(type):
        def __hash__(self):
            if is_hashable:
                return 1
            raise TypeError("not hashable")

    class A(metaclass=UnhashableMeta):
        pass

    class B(metaclass=UnhashableMeta):
        pass

    union = A | B
    assert union.__args__ == (A, B)
    with assert_raises(TypeError) as raised:
        hash(union)
    assert str(raised.exception) == "not hashable"

    is_hashable = True
    with assert_raises(TypeError) as raised:
        hash(union)
    assert str(raised.exception) == "union contains 2 unhashable elements"


_union_unhashable_partition_is_stable()


def _run_missing_type_params_regression():
    args = _ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    pass_stmt = _ast.Pass(lineno=1, col_offset=4, end_lineno=1, end_col_offset=8)
    fn = _ast.FunctionDef("f", args, [pass_stmt], [], None, None)
    fn.lineno = 1
    fn.col_offset = 0
    fn.end_lineno = 1
    fn.end_col_offset = 8
    mod = _ast.Module([fn], [])
    compiled = compile(mod, "<stdlib_types_missing_type_params>", "exec")
    exec(compiled, {})


_run_missing_type_params_regression()
