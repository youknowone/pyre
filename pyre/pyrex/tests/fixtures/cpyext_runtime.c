/* The runtime services an extension reaches for around a call: the exception
   a failed syscall becomes, the audit events it raises, importing one
   attribute out of a module, and writing a name into a module's namespace. */

#include <Python.h>

#include <errno.h>
#include <string.h>

/* The pending exception's class name and message, taken and cleared. */
static PyObject *pending(void)
{
    PyObject *value = PyErr_GetRaisedException();
    if (value == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *text = PyObject_Str(value);
    PyObject *pair = Py_BuildValue("(sO)", Py_TYPE(value)->tp_name,
                                   text == NULL ? Py_None : text);
    Py_XDECREF(text);
    Py_DECREF(value);
    return pair;
}

/* ── the failed syscall ───────────────────────────────────────────────── */

/* The exception `PyErr_SetFromErrno` and its filename spellings raise, taken
   apart into the pieces `OSError` files them under.  `which` picks the
   spelling; `code` is stamped into `errno` first, which is what a real syscall
   would have left there. */
static PyObject *from_errno(PyObject *self, PyObject *args)
{
    (void)self;
    const char *which;
    int code;
    PyObject *klass;
    PyObject *first;
    PyObject *second;
    if (!PyArg_ParseTuple(args, "siOOO", &which, &code, &klass, &first, &second)) {
        return NULL;
    }
    PyObject *answer;
    errno = code;
    if (strcmp(which, "plain") == 0) {
        answer = PyErr_SetFromErrno(klass);
    } else if (strcmp(which, "filename") == 0) {
        const char *name = first == Py_None ? NULL : PyUnicode_AsUTF8(first);
        answer = PyErr_SetFromErrnoWithFilename(klass, name);
    } else if (strcmp(which, "object") == 0) {
        answer = PyErr_SetFromErrnoWithFilenameObject(
            klass, first == Py_None ? NULL : first);
    } else if (strcmp(which, "objects") == 0) {
        answer = PyErr_SetFromErrnoWithFilenameObjects(
            klass, first == Py_None ? NULL : first,
            second == Py_None ? NULL : second);
    } else {
        PyErr_SetString(PyExc_ValueError, "the fixture does not offer that spelling");
        return NULL;
    }
    if (answer != NULL) {
        Py_DECREF(answer);
        PyErr_SetString(PyExc_AssertionError, "the entry point answered non-NULL");
        return NULL;
    }
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *fields = PyObject_GetAttrString(raised, "args");
    PyObject *text = PyObject_Str(raised);
    PyObject *row = Py_BuildValue("(sOO)", Py_TYPE(raised)->tp_name,
                                  fields == NULL ? Py_None : fields,
                                  text == NULL ? Py_None : text);
    Py_XDECREF(fields);
    Py_XDECREF(text);
    Py_DECREF(raised);
    return row;
}

/* `PyErr_CheckSignals` with nothing pending, which is the only state a test
   can arrange without sending itself a signal. */
static PyObject *check_signals(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    int answer = PyErr_CheckSignals();
    return Py_BuildValue("(iO)", answer, pending());
}

/* ── the audit events ─────────────────────────────────────────────────── */

/* Raise an event through the variadic spelling, under a format the caller
   names.  Each shape is one the header's builder has to reach a tuple by a
   different route. */
static PyObject *audit(PyObject *self, PyObject *args)
{
    (void)self;
    const char *event;
    const char *shape;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "ssO", &event, &shape, &value)) {
        return NULL;
    }
    int answer;
    if (strcmp(shape, "none") == 0) {
        answer = PySys_Audit(event, NULL);
    } else if (strcmp(shape, "empty") == 0) {
        answer = PySys_Audit(event, "");
    } else if (strcmp(shape, "one") == 0) {
        answer = PySys_Audit(event, "O", value);
    } else if (strcmp(shape, "two") == 0) {
        answer = PySys_Audit(event, "Oi", value, 7);
    } else if (strcmp(shape, "int") == 0) {
        answer = PySys_Audit(event, "i", 42);
    } else if (strcmp(shape, "string") == 0) {
        answer = PySys_Audit(event, "s", "text");
    } else {
        PyErr_SetString(PyExc_ValueError, "the fixture does not offer that shape");
        return NULL;
    }
    return Py_BuildValue("(iO)", answer, pending());
}

/* The tuple spelling, including what a non-tuple argument is answered with. */
static PyObject *audit_tuple(PyObject *self, PyObject *args)
{
    (void)self;
    const char *event;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "sO", &event, &value)) {
        return NULL;
    }
    int answer = PySys_AuditTuple(event, value == Py_None ? NULL : value);
    return Py_BuildValue("(iO)", answer, pending());
}

/* ── one attribute of a module ────────────────────────────────────────── */

static PyObject *module_attr(PyObject *self, PyObject *args)
{
    (void)self;
    const char *module;
    const char *attribute;
    if (!PyArg_ParseTuple(args, "ss", &module, &attribute)) {
        return NULL;
    }
    PyObject *value = PyImport_ImportModuleAttrString(module, attribute);
    if (value == NULL) {
        return Py_BuildValue("(OO)", Py_None, pending());
    }
    PyObject *row = Py_BuildValue("(OO)", value, Py_None);
    Py_DECREF(value);
    return row;
}

static PyObject *module_attr_object(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *module;
    PyObject *attribute;
    if (!PyArg_ParseTuple(args, "OO", &module, &attribute)) {
        return NULL;
    }
    PyObject *value = PyImport_ImportModuleAttr(module, attribute);
    if (value == NULL) {
        return Py_BuildValue("(OO)", Py_None, pending());
    }
    PyObject *row = Py_BuildValue("(OO)", value, Py_None);
    Py_DECREF(value);
    return row;
}

/* ── a name written into a module's namespace ───────────────── */

/* `PyModule_AddObjectRef(target, name, value)`, where the caller names the
   target rather than being handed a module, so a non-module reaches it. */
static PyObject *add_object_ref(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *target;
    const char *name;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "OsO", &target, &name, &value)) {
        return NULL;
    }
    int answer = PyModule_AddObjectRef(target, name, value);
    return Py_BuildValue("(iN)", answer, pending());
}

/* The same with a NULL value, which is how a caller reports that the object it
   meant to add could not be built.  `raise` says whether it left an exception
   behind first, which is what the entry point is documented to require. */
static PyObject *add_object_ref_null(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *target;
    const char *name;
    int raise;
    if (!PyArg_ParseTuple(args, "Osi", &target, &name, &raise)) {
        return NULL;
    }
    if (raise) {
        PyErr_SetString(PyExc_KeyError, "the caller's own reason");
    }
    int answer = PyModule_AddObjectRef(target, name, NULL);
    return Py_BuildValue("(iN)", answer, pending());
}

static PyMethodDef methods[] = {
    {"from_errno", from_errno, METH_VARARGS, NULL},
    {"check_signals", check_signals, METH_NOARGS, NULL},
    {"audit", audit, METH_VARARGS, NULL},
    {"audit_tuple", audit_tuple, METH_VARARGS, NULL},
    {"module_attr", module_attr, METH_VARARGS, NULL},
    {"module_attr_object", module_attr_object, METH_VARARGS, NULL},
    {"add_object_ref", add_object_ref, METH_VARARGS, NULL},
    {"add_object_ref_null", add_object_ref_null, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_runtime", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_runtime(void)
{
    PyObject *module = PyModule_Create(&def);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "ENOENT", (long)ENOENT) < 0 ||
        PyModule_AddIntConstant(module, "EPERM", (long)EPERM) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
