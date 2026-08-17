//! _functools module — CPython accelerator imported by
//! `lib-python/3/functools.py`.
//!
//! `cmp_to_key` follows the stdlib fallback structurally: each invocation
//! creates a lexical `K`, capturing `mycmp` instead of exposing it on K.
//! `W_KeyWrapper` below carries the `keyobject` payload shape but is not what
//! `cmp_to_key` hands out — its `__call__` is generated positional-only, while
//! `keyobject` takes `obj` by keyword and reports `(obj)` to `inspect`.
//! `partial` follows PyPy's `lib_pypy/_functools.py`: its public state is
//! exposed through read-only properties backed by private slots. Placeholder
//! argument merging follows the Python 3.14 `functools.py` implementation.

use pyre_object::*;

/// `_functoolsmodule.c keyobject`.  The object returned by `cmp_to_key` is
/// itself the one-argument key factory; calling it produces another wrapper
/// carrying the value to compare.  Keeping both references on the payload
/// mirrors `keyobject.cmp` / `keyobject.object` and, unlike an app-level
/// function, leaves the module callable descriptor-neutral when a user stores
/// it on a class.
#[crate::pyre_class("functools.KeyWrapper")]
pub struct W_KeyWrapper {
    cmp: PyObjectRef,
    object: PyObjectRef,
}

fn key_wrapper_new(cmp: PyObjectRef, object: PyObjectRef) -> PyObjectRef {
    // The factory is called directly (rather than through `type.__call__`),
    // so force the TypeDef/typed-layout binding before allocating its payload.
    let _ = type_object();
    let _roots = pyre_object::gc_roots::push_roots();
    // Publish both livevars before the first forwarding query: `pin_root` can
    // park behind another thread's collection, which would leave the operand
    // still held in a Rust local naming a pre-move address.
    let slot = pyre_object::gc_roots::pin_roots(&[cmp, object]);
    W_KeyWrapper::allocate_stable(W_KeyWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        cmp: pyre_object::gc_roots::shadow_stack_get(slot),
        object: pyre_object::gc_roots::shadow_stack_get(slot + 1),
    })
}

fn key_wrapper_compare(
    this: &W_KeyWrapper,
    other: PyObjectRef,
    op: crate::baseobjspace::CompareOp,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(other) = W_KeyWrapper::from_obj(other) else {
        return Err(crate::PyError::type_error(
            "other argument must be K instance",
        ));
    };
    let _roots = pyre_object::gc_roots::push_roots();
    // `this` and `other` name payloads in the managed heap, so every operand
    // has to reach the shadow stack before the first forwarding query.
    let slot = pyre_object::gc_roots::pin_roots(&[this.cmp, this.object, other.object]);
    let result = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(slot + 1),
            pyre_object::gc_roots::shadow_stack_get(slot + 2),
        ],
    )?;
    pyre_object::gc_roots::pin_root(result);
    // Build the comparison operand first: a custom numeric result makes
    // `w_int_new` allocate, and reading the result slot before that would leave
    // the argument naming a pre-move address.
    let w_zero = w_int_new(0);
    crate::baseobjspace::compare(
        pyre_object::gc_roots::shadow_stack_get(slot + 3),
        w_zero,
        op,
    )
}

#[crate::pyre_methods(unhashable)]
impl W_KeyWrapper {
    fn __call__(&self, obj: PyObjectRef) -> PyObjectRef {
        key_wrapper_new(self.cmp, obj)
    }

    #[getter]
    fn obj(&self) -> PyObjectRef {
        self.object
    }

    fn __lt__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        key_wrapper_compare(self, other, crate::baseobjspace::CompareOp::Lt)
    }
    fn __le__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        key_wrapper_compare(self, other, crate::baseobjspace::CompareOp::Le)
    }
    fn __eq__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        key_wrapper_compare(self, other, crate::baseobjspace::CompareOp::Eq)
    }
    fn __ne__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        key_wrapper_compare(self, other, crate::baseobjspace::CompareOp::Ne)
    }
    fn __gt__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        key_wrapper_compare(self, other, crate::baseobjspace::CompareOp::Gt)
    }
    fn __ge__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        key_wrapper_compare(self, other, crate::baseobjspace::CompareOp::Ge)
    }
}

/// PyPy `module/_functools/interp_functools.py:reduce`.
///
/// Keep the accumulator in a GC-walked one-element list: `next()` and the
/// reduction callback may both collect, so a Rust local would not track a
/// relocated object between iterations.
fn reduce(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let keyword_initial =
        kwargs.and_then(|dict| unsafe { pyre_object::w_dict_getitem_str(dict, "initial") });
    let keyword_count = crate::builtins::real_kwarg_count(kwargs);
    if keyword_count != usize::from(keyword_initial.is_some()) {
        return Err(crate::PyError::type_error(
            "reduce() got an unexpected keyword argument",
        ));
    }
    // CPython 3.14 `_functoolsmodule.c:reduce` keeps `function` and
    // `iterable` positional-only even though `initial` may be named.  A named
    // initial cannot fill either required position.
    if positional.len() < 2 {
        return Err(crate::PyError::type_error(format!(
            "reduce() takes at least 2 positional arguments ({} given)",
            positional.len()
        )));
    }
    let mut effective = positional.to_vec();
    if let Some(initial) = keyword_initial {
        effective.push(initial);
    }
    if effective.len() > 3 {
        return Err(crate::PyError::type_error(format!(
            "reduce() takes at most 3 arguments ({} given)",
            effective.len()
        )));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    // Publish every operand before the first forwarding query: a `pin_root`
    // per argument can park behind another thread's collection, leaving the
    // ones still in `effective` naming pre-move addresses.
    let base = pyre_object::gc_roots::pin_roots(&effective);
    let function_slot = base;
    let sequence_slot = base + 1;
    // `_functoolsmodule.c:reduce` reports the failed second argument itself
    // rather than letting the iteration protocol's own message surface.
    let w_iter =
        match crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(sequence_slot)) {
            Ok(w_iter) => w_iter,
            Err(err) if err.kind == crate::PyErrorKind::TypeError => {
                return Err(crate::PyError::type_error(
                    "reduce() arg 2 must support iteration",
                ));
            }
            Err(err) => return Err(err),
        };
    pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let initial = if effective.len() == 3 {
        pyre_object::gc_roots::shadow_stack_get(base + 2)
    } else {
        match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot)) {
            Ok(value) => value,
            Err(err) if err.matches_stop_iteration() => {
                return Err(crate::PyError::type_error(
                    "reduce() of empty iterable with no initial value",
                ));
            }
            Err(err) => return Err(err),
        }
    };
    pyre_object::gc_roots::pin_root(initial);
    let initial_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let accumulator =
        pyre_object::listobject::w_list_new(vec![pyre_object::gc_roots::shadow_stack_get(
            initial_slot,
        )]);
    pyre_object::gc_roots::pin_root(accumulator);
    let accumulator_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    loop {
        let item =
            match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot)) {
                Ok(value) => value,
                Err(err) if err.matches_stop_iteration() => break,
                Err(err) => return Err(err),
            };
        // `next()` returns a raw object reference.  Keep this iteration's
        // transient values in their own root scope: the reducer is arbitrary
        // Python and `w_list_setitem` may switch list strategy and allocate.
        let _iteration_roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(item);
        let item_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let current = unsafe {
            pyre_object::listobject::w_list_getitem(
                pyre_object::gc_roots::shadow_stack_get(accumulator_slot),
                0,
            )
            .unwrap()
        };
        let result = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(function_slot),
            &[current, pyre_object::gc_roots::shadow_stack_get(item_slot)],
        )?;
        pyre_object::gc_roots::pin_root(result);
        let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            pyre_object::listobject::w_list_setitem(
                pyre_object::gc_roots::shadow_stack_get(accumulator_slot),
                0,
                pyre_object::gc_roots::shadow_stack_get(result_slot),
            );
        }
    }
    Ok(unsafe {
        pyre_object::listobject::w_list_getitem(
            pyre_object::gc_roots::shadow_stack_get(accumulator_slot),
            0,
        )
        .unwrap()
    })
}

crate::py_module! {
    "_functools",
    inline_app: {
        r#"
from operator import itemgetter as _partial_itemgetter
from reprlib import recursive_repr as _partial_recursive_repr
from types import GenericAlias as _PartialGenericAlias
from types import MethodType as _PartialMethodType


def _placeholder_immutable(name):
    return TypeError(
        f"cannot set {name!r} attribute of immutable type "
        "'functools._PlaceholderType'"
    )


class _PlaceholderMeta(type):
    # The accelerator's type is static, so `type_setattro` refuses both an
    # assignment and a deletion, with the same wording.  The pure-Python
    # fallback in `functools.py` is an ordinary class and accepts them; this
    # module stands in for the accelerator.
    def __setattr__(cls, name, value):
        raise _placeholder_immutable(name)

    def __delattr__(cls, name):
        raise _placeholder_immutable(name)


# The singleton cannot live on the class, whose attributes are now read-only.
_placeholder_instance = None


class _PlaceholderType(metaclass=_PlaceholderMeta):
    """The type of the Placeholder singleton."""

    __module__ = "functools"
    __slots__ = ()

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "type 'functools._PlaceholderType' is not an acceptable base type"
        )

    def __new__(cls):
        global _placeholder_instance
        if _placeholder_instance is None:
            _placeholder_instance = object.__new__(cls)
        return _placeholder_instance

    def __repr__(self):
        return "Placeholder"

    def __reduce__(self):
        return "Placeholder"


Placeholder = _PlaceholderType()


def _partial_prepare_merger(args):
    if not args:
        return 0, None
    nargs = len(args)
    order = []
    j = nargs
    for i, arg in enumerate(args):
        if arg is Placeholder:
            order.append(j)
            j += 1
        else:
            order.append(i)
    phcount = j - nargs
    merger = _partial_itemgetter(*order) if phcount else None
    return phcount, merger


def _partial_new(cls, func, /, *args, **keywords):
    if not callable(func):
        raise TypeError("the first argument must be callable")
    if args and args[-1] is Placeholder:
        raise TypeError("trailing Placeholders are not allowed")
    for value in keywords.values():
        if value is Placeholder:
            raise TypeError("Placeholder cannot be passed as a keyword argument")

    # `partial_new` unpacks a wrapped partial only while it carries no
    # instance state (`part->dict == NULL`); placeholder positions and bound
    # arguments are merged for subclasses too, since subclassing alone does
    # not block the optimization.  The slots above keep `__dict__` empty until
    # user code assigns an attribute, so a partial holding its own state stays
    # wrapped and keeps running its own behaviour.
    if isinstance(func, partial) and not func.__dict__:
        pto_phcount = func._phcount
        tot_args = func.args
        if args:
            tot_args += args
            if pto_phcount:
                nargs = len(args)
                if nargs < pto_phcount:
                    tot_args += (Placeholder,) * (pto_phcount - nargs)
                tot_args = func._merger(tot_args)
                if nargs > pto_phcount:
                    tot_args += args[pto_phcount:]
            phcount, merger = _partial_prepare_merger(tot_args)
        else:
            phcount, merger = pto_phcount, func._merger
        keywords = {**func.keywords, **keywords}
        func = func.func
    else:
        tot_args = args
        phcount, merger = _partial_prepare_merger(tot_args)

    self = object.__new__(cls)
    object.__setattr__(self, "_func", func)
    object.__setattr__(self, "_args", tot_args)
    object.__setattr__(self, "_keywords", keywords)
    object.__setattr__(self, "_phcount", phcount)
    object.__setattr__(self, "_merger", merger)
    return self


def _partial_repr(self):
    cls = type(self)
    func, p_args, keywords = self.func, self.args, self.keywords
    args = [repr(func)]
    args.extend(map(repr, p_args))
    args.extend(f"{key}={value!r}" for key, value in keywords.items())
    return f"{cls.__module__}.{cls.__qualname__}({', '.join(args)})"


class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    # CPython's accelerator keeps these three fields in the partial object and
    # exposes them through read-only member descriptors.  Private slots plus
    # getter-only properties preserve the same ownership and mutability shape.
    __slots__ = (
        "_func", "_args", "_keywords", "_phcount", "_merger",
        "__dict__", "__weakref__",
    )

    __new__ = _partial_new
    __repr__ = _partial_recursive_repr()(_partial_repr)

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def keywords(self):
        return self._keywords

    def __delattr__(self, name):
        if name == "__dict__":
            raise TypeError("a partial object's dictionary may not be deleted")
        object.__delattr__(self, name)

    def __call__(self, /, *args, **keywords):
        phcount = self._phcount
        if phcount:
            try:
                pto_args = self._merger(self.args + args)
                args = args[phcount:]
            except IndexError:
                raise TypeError(
                    "missing positional arguments in 'partial' call; "
                    f"expected at least {phcount}, got {len(args)}"
                )
        else:
            pto_args = self.args
        keywords = {**self.keywords, **keywords}
        return self.func(*pto_args, *args, **keywords)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _PartialMethodType(self, obj)

    def __reduce__(self):
        return (
            type(self),
            (self.func,),
            (self.func, self.args, self.keywords or None, self.__dict__ or None),
        )

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (
            not callable(func)
            or not isinstance(args, tuple)
            or (kwds is not None and not isinstance(kwds, dict))
            or (namespace is not None and not isinstance(namespace, dict))
        ):
            raise TypeError("invalid partial state")
        if args and args[-1] is Placeholder:
            raise TypeError("trailing Placeholders are not allowed")

        phcount, merger = _partial_prepare_merger(args)
        args = tuple(args)
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_keywords", kwds)
        object.__setattr__(self, "_phcount", phcount)
        object.__setattr__(self, "_merger", merger)

    __class_getitem__ = classmethod(_PartialGenericAlias)


partial.__module__ = "functools"


def cmp_to_key(mycmp):
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

# `_functools.cmp_to_key` is an interp-level builtin in CPython.  Unlike an
# app-level function, it therefore does not acquire an instance when a caller
# stores it on a class (the CPython functools tests do exactly that).  A
# callable staticmethod preserves the app-level implementation while giving
# the exported object the same non-binding descriptor behavior.
cmp_to_key = staticmethod(cmp_to_key)
"# => ["cmp_to_key", "partial", "Placeholder", "_PlaceholderType"],
    },
    functions: {
        "reduce" / * = reduce,
    },
}
