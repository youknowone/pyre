/* The protocols that work on any object, and the call interface.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_ABSTRACT_H
#define PYRE_ABSTRACT_H

#ifdef __cplusplus
extern "C" {
#endif
/* The vectorcall protocol.  `args[0 .. nargs)` are positional and the rest are
   the values for the names in `kwnames`, in order.  The high bit of `nargsf`
   says the caller left a spare slot before args[0]; pyre only reads the array,
   so the bit just has to be stripped from the count. */
#define PY_VECTORCALL_ARGUMENTS_OFFSET ((size_t)1 << (8 * sizeof(size_t) - 1))

static inline Py_ssize_t PyVectorcall_NARGS(size_t n)
{
    return (Py_ssize_t)(n & ~PY_VECTORCALL_ARGUMENTS_OFFSET);
}

static inline PyObject *PyObject_CallMethodNoArgs(PyObject *self, PyObject *name)
{
    PyObject *args[1];
    args[0] = self;
    return PyObject_VectorcallMethod(name, args, 1, NULL);
}

static inline PyObject *PyObject_CallMethodOneArg(PyObject *self, PyObject *name,
                                                  PyObject *arg)
{
    PyObject *args[2];
    args[0] = self;
    args[1] = arg;
    return PyObject_VectorcallMethod(name, args, 2, NULL);
}

/* The number protocol. */

/* The sequence protocol. */

/* Functions rather than the macros the reference header spells: a mirror has
   no item array of its own, so the length and the items come from the
   interpreter object behind it. */

/* The mapping protocol. */

/* The iterator protocol. */

/* The `ObjArgs` pair take a NULL-terminated argument list.  The list is walked
   twice -- once to count and once to fill a tuple -- rather than collected into
   a fixed buffer, so an arity no buffer anticipated still works. */
static inline PyObject *_PyPyre_ObjArgsTuple(va_list count_va, va_list fill_va)
{
    Py_ssize_t size = 0;
    while (va_arg(count_va, PyObject *) != NULL) {
        size++;
    }
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        PyObject *item = va_arg(fill_va, PyObject *);
        Py_XINCREF(item);
        PyTuple_SetItem(tuple, index, item);
    }
    return tuple;
}

static inline PyObject *PyObject_CallFunctionObjArgs(PyObject *callable, ...)
{
    va_list count_va, fill_va;
    va_start(count_va, callable);
    va_start(fill_va, callable);
    PyObject *args = _PyPyre_ObjArgsTuple(count_va, fill_va);
    va_end(count_va);
    va_end(fill_va);
    if (args == NULL) {
        return NULL;
    }
    PyObject *result = PyObject_Call(callable, args, NULL);
    Py_DECREF(args);
    return result;
}

static inline PyObject *PyObject_CallMethodObjArgs(PyObject *self, PyObject *name, ...)
{
    PyObject *method = PyObject_GetAttr(self, name);
    if (method == NULL) {
        return NULL;
    }
    va_list count_va, fill_va;
    va_start(count_va, name);
    va_start(fill_va, name);
    PyObject *args = _PyPyre_ObjArgsTuple(count_va, fill_va);
    va_end(count_va);
    va_end(fill_va);
    if (args == NULL) {
        Py_DECREF(method);
        return NULL;
    }
    PyObject *result = PyObject_Call(method, args, NULL);
    Py_DECREF(args);
    Py_DECREF(method);
    return result;
}

/* `PyObject_CallFunction` and `PyObject_CallMethod` take a `Py_BuildValue`
   format.  A format building exactly one value that is already a tuple is the
   argument list itself; anything else becomes a one-element list. */
static inline PyObject *_PyPyre_CallWithFormat(PyObject *callable, PyObject *built)
{
    if (built == NULL) {
        return NULL;
    }
    PyObject *args = built;
    if (!PyTuple_Check(built)) {
        args = PyTuple_New(1);
        if (args == NULL) {
            Py_DECREF(built);
            return NULL;
        }
        PyTuple_SetItem(args, 0, built);   /* steals `built` */
    }
    PyObject *result = PyObject_Call(callable, args, NULL);
    Py_DECREF(args);
    return result;
}

static inline PyObject *PyObject_CallFunction(PyObject *callable, const char *format, ...)
{
    if (format == NULL || *format == '\0') {
        return PyObject_CallNoArgs(callable);
    }
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *built = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    return _PyPyre_CallWithFormat(callable, built);
}

static inline PyObject *PyObject_CallMethod(PyObject *self, const char *name,
                                            const char *format, ...)
{
    PyObject *method = PyObject_GetAttrString(self, name);
    if (method == NULL) {
        return NULL;
    }
    if (format == NULL || *format == '\0') {
        PyObject *result = PyObject_CallNoArgs(method);
        Py_DECREF(method);
        return result;
    }
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *built = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    PyObject *result = _PyPyre_CallWithFormat(method, built);
    Py_DECREF(method);
    return result;
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_ABSTRACT_H */
