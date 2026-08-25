/* The entry points an extension reaches when it declares the prototypes
   itself instead of expanding this header's macros.

   PyO3 and Cython both generate their own declarations of CPython's ABI and
   never read these headers, so a name pyre only ever spells as a macro is a
   name their objects import and `dlopen` cannot resolve.  Everything here is
   called as a function for that reason, `Py_TYPE` with the macro undefined so
   that the call is the export rather than the field read. */

#include <Python.h>

#undef Py_TYPE

/* Whether the reference count moves the same way through both spellings of
   the pair -- the macro's, and the stable ABI's own entry points. */
static PyObject *refcount_through_both(PyObject *self, PyObject *value)
{
    (void)self;
    Py_ssize_t start = Py_REFCNT(value);
    Py_IncRef(value);
    Py_ssize_t after_public = Py_REFCNT(value);
    Py_DecRef(value);
    _Py_IncRef(value);
    Py_ssize_t after_stable = Py_REFCNT(value);
    _Py_DecRef(value);
    return Py_BuildValue("(nnn)", start, after_public - start,
                         after_stable - start);
}

/* `Py_TYPE` as a call answers the same block the macro reads. */
static PyObject *type_through_call(PyObject *self, PyObject *value)
{
    (void)self;
    PyTypeObject *called = Py_TYPE(value);
    PyTypeObject *field = ((PyObject *)value)->ob_type;
    return Py_BuildValue("(sO)", called->tp_name,
                         called == field ? Py_True : Py_False);
}

/* The two questions an extension asks about the interpreter that loaded it. */
static PyObject *runtime_state(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    return Py_BuildValue("(ii)", Py_IsInitialized(), Py_IsFinalizing());
}

/* A critical section around a call, entered and left by name. */
static PyObject *inside_critical_section(PyObject *self, PyObject *callable)
{
    (void)self;
    PyCriticalSection section;
    PyCriticalSection_Begin(&section, callable);
    PyObject *answer = PyObject_CallNoArgs(callable);
    PyCriticalSection_End(&section);
    return answer;
}

/* `PyErr_PrintEx` reports through `sys.excepthook` and clears the indicator;
   with the flag set it records the exception on `sys` first. */
static PyObject *print_pending(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value;
    int set_sys_last_vars;
    if (!PyArg_ParseTuple(args, "Oi", &value, &set_sys_last_vars)) {
        return NULL;
    }
    PyErr_SetObject((PyObject *)Py_TYPE(value), value);
    PyErr_PrintEx(set_sys_last_vars);
    return Py_BuildValue("O", PyErr_Occurred() == NULL ? Py_True : Py_False);
}

/* `PyTraceBack_Print` writes the header line and then the entries. */
static PyObject *print_traceback(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *traceback;
    PyObject *file;
    if (!PyArg_ParseTuple(args, "OO", &traceback, &file)) {
        return NULL;
    }
    int printed = PyTraceBack_Print(traceback, file);
    /* The refusal path leaves an indicator, and this returns a value rather
       than NULL, so it is taken here instead of following the answer out. */
    PyErr_Clear();
    return Py_BuildValue("i", printed);
}

/* The exception a decoder raises, built from the bytes it was decoding --
   embedded NULs included, which is why the length is passed separately. */
static PyObject *decode_error(PyObject *self, PyObject *args)
{
    (void)self;
    const char *encoding;
    const char *object;
    Py_ssize_t length;
    Py_ssize_t start;
    Py_ssize_t end;
    const char *reason;
    if (!PyArg_ParseTuple(args, "sy#nns", &encoding, &object, &length, &start,
                          &end, &reason)) {
        return NULL;
    }
    return PyUnicodeDecodeError_Create(encoding, object, length, start, end,
                                       reason);
}

static PyMethodDef methods[] = {
    {"refcount_through_both", refcount_through_both, METH_O, NULL},
    {"type_through_call", type_through_call, METH_O, NULL},
    {"runtime_state", runtime_state, METH_NOARGS, NULL},
    {"inside_critical_section", inside_critical_section, METH_O, NULL},
    {"print_pending", print_pending, METH_VARARGS, NULL},
    {"print_traceback", print_traceback, METH_VARARGS, NULL},
    {"decode_error", decode_error, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_stable_abi",
                                 NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_stable_abi(void)
{
    return PyModule_Create(&def);
}
