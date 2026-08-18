/* The warnings entry points.

   Each function answers the return value and leaves whatever the machinery
   did behind, so a Python-side comparison against CPython running the same
   code says whether the two agree. */

#include <Python.h>

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

/* The return value and whatever was left pending, which is the pair every row
   below is compared by. */
static PyObject *outcome(int returned)
{
    PyObject *left = pending();
    PyObject *answer = Py_BuildValue("(iO)", returned, left == NULL ? Py_None : left);
    Py_XDECREF(left);
    return answer;
}

static PyObject *warn_ex(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category;
    const char *message;
    Py_ssize_t stack_level;
    if (!PyArg_ParseTuple(args, "Osn", &category, &message, &stack_level)) {
        return NULL;
    }
    return outcome(PyErr_WarnEx(category == Py_None ? NULL : category, message,
                                stack_level));
}

/* A message that is not valid UTF-8, which the `s` format could not carry. */
static PyObject *warn_ex_bytes(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category;
    const char *message;
    Py_ssize_t length, stack_level;
    if (!PyArg_ParseTuple(args, "Oy#n", &category, &message, &length, &stack_level)) {
        return NULL;
    }
    return outcome(PyErr_WarnEx(category == Py_None ? NULL : category, message,
                                stack_level));
}

static PyObject *warn_format(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category, *object;
    Py_ssize_t stack_level;
    if (!PyArg_ParseTuple(args, "OnO", &category, &stack_level, &object)) {
        return NULL;
    }
    return outcome(PyErr_WarnFormat(category == Py_None ? NULL : category, stack_level,
                                    "s=%s d=%d o=%R", "abc", 42, object));
}

static PyObject *resource_warning(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *source;
    Py_ssize_t stack_level;
    if (!PyArg_ParseTuple(args, "On", &source, &stack_level)) {
        return NULL;
    }
    return outcome(PyErr_ResourceWarning(source == Py_None ? NULL : source, stack_level,
                                         "unclosed %s at %d", "sock", 4));
}

/* The deprecated macro, which is `PyErr_WarnEx` with a stack level of 1. */
static PyObject *warn_macro(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category;
    const char *message;
    if (!PyArg_ParseTuple(args, "Os", &category, &message)) {
        return NULL;
    }
    return outcome(PyErr_Warn(category == Py_None ? NULL : category, message));
}

static PyObject *warn_explicit(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category, *registry;
    const char *message, *filename, *module;
    int lineno;
    if (!PyArg_ParseTuple(args, "OssizO", &category, &message, &filename, &lineno,
                          &module, &registry)) {
        return NULL;
    }
    return outcome(PyErr_WarnExplicit(category == Py_None ? NULL : category, message,
                                      filename, lineno, module,
                                      registry == Py_None ? NULL : registry));
}

static PyObject *warn_explicit_object(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category, *message, *filename, *module, *registry;
    int lineno;
    if (!PyArg_ParseTuple(args, "OOOiOO", &category, &message, &filename, &lineno,
                          &module, &registry)) {
        return NULL;
    }
    return outcome(PyErr_WarnExplicitObject(category, message, filename, lineno,
                                            module, registry));
}

static PyObject *warn_explicit_format(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *category, *registry;
    const char *filename, *module;
    int lineno;
    if (!PyArg_ParseTuple(args, "OsizO", &category, &filename, &lineno, &module,
                          &registry)) {
        return NULL;
    }
    return outcome(PyErr_WarnExplicitFormat(category == Py_None ? NULL : category,
                                            filename, lineno, module,
                                            registry == Py_None ? NULL : registry,
                                            "s=%s d=%d", "xyz", -7));
}

/* Every optional argument passed as NULL, which the pass-through form above
   cannot spell. */
static PyObject *warn_explicit_object_null(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *message, *filename;
    int lineno;
    if (!PyArg_ParseTuple(args, "OOi", &message, &filename, &lineno)) {
        return NULL;
    }
    return outcome(PyErr_WarnExplicitObject(NULL, message, filename, lineno, NULL,
                                            NULL));
}

/* Every `PyExc_*` an extension can link against, paired with the class it
   resolves to.  A symbol that never got bound reports `<null>`. */
static int add_mirror(PyObject *answer, const char *symbol, PyObject *mirror)
{
    PyObject *item = Py_BuildValue("(ss)", symbol,
                                   mirror == NULL ? "<null>"
                                                  : ((PyTypeObject *)mirror)->tp_name);
    if (item == NULL) {
        return -1;
    }
    int appended = PyList_Append(answer, item);
    Py_DECREF(item);
    return appended;
}

#define MIRROR(name)                                       \
    if (add_mirror(answer, #name, PyExc_##name) < 0) {     \
        Py_DECREF(answer);                                 \
        return NULL;                                       \
    }

static PyObject *mirrors(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    PyObject *answer = PyList_New(0);
    if (answer == NULL) {
        return NULL;
    }
    MIRROR(BaseException)
    MIRROR(Exception)
    MIRROR(ArithmeticError)
    MIRROR(AssertionError)
    MIRROR(AttributeError)
    MIRROR(BaseExceptionGroup)
    MIRROR(BlockingIOError)
    MIRROR(BrokenPipeError)
    MIRROR(BufferError)
    MIRROR(ChildProcessError)
    MIRROR(ConnectionAbortedError)
    MIRROR(ConnectionError)
    MIRROR(ConnectionRefusedError)
    MIRROR(ConnectionResetError)
    MIRROR(EOFError)
    MIRROR(FileExistsError)
    MIRROR(FileNotFoundError)
    MIRROR(FloatingPointError)
    MIRROR(GeneratorExit)
    MIRROR(ImportError)
    MIRROR(IndentationError)
    MIRROR(IndexError)
    MIRROR(InterruptedError)
    MIRROR(IsADirectoryError)
    MIRROR(KeyError)
    MIRROR(KeyboardInterrupt)
    MIRROR(LookupError)
    MIRROR(MemoryError)
    MIRROR(ModuleNotFoundError)
    MIRROR(NameError)
    MIRROR(NotADirectoryError)
    MIRROR(NotImplementedError)
    MIRROR(OSError)
    MIRROR(OverflowError)
    MIRROR(PermissionError)
    MIRROR(ProcessLookupError)
    MIRROR(PythonFinalizationError)
    MIRROR(RecursionError)
    MIRROR(ReferenceError)
    MIRROR(RuntimeError)
    MIRROR(StopAsyncIteration)
    MIRROR(StopIteration)
    MIRROR(SyntaxError)
    MIRROR(SystemError)
    MIRROR(SystemExit)
    MIRROR(TabError)
    MIRROR(TimeoutError)
    MIRROR(TypeError)
    MIRROR(UnboundLocalError)
    MIRROR(UnicodeDecodeError)
    MIRROR(UnicodeEncodeError)
    MIRROR(UnicodeError)
    MIRROR(UnicodeTranslateError)
    MIRROR(ValueError)
    MIRROR(ZeroDivisionError)
    MIRROR(EnvironmentError)
    MIRROR(IOError)
    MIRROR(Warning)
    MIRROR(BytesWarning)
    MIRROR(DeprecationWarning)
    MIRROR(EncodingWarning)
    MIRROR(FutureWarning)
    MIRROR(ImportWarning)
    MIRROR(PendingDeprecationWarning)
    MIRROR(ResourceWarning)
    MIRROR(RuntimeWarning)
    MIRROR(SyntaxWarning)
    MIRROR(UnicodeWarning)
    MIRROR(UserWarning)
    return answer;
}

static PyMethodDef methods[] = {
    {"warn_ex", warn_ex, METH_VARARGS, NULL},
    {"warn_ex_bytes", warn_ex_bytes, METH_VARARGS, NULL},
    {"warn_format", warn_format, METH_VARARGS, NULL},
    {"resource_warning", resource_warning, METH_VARARGS, NULL},
    {"warn_macro", warn_macro, METH_VARARGS, NULL},
    {"warn_explicit", warn_explicit, METH_VARARGS, NULL},
    {"warn_explicit_object", warn_explicit_object, METH_VARARGS, NULL},
    {"warn_explicit_object_null", warn_explicit_object_null, METH_VARARGS, NULL},
    {"mirrors", mirrors, METH_NOARGS, NULL},
    {"warn_explicit_format", warn_explicit_format, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_warnings", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_warnings(void)
{
    return PyModule_Create(&def);
}
