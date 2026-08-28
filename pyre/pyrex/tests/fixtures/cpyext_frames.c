/* The frame an extension builds for itself, and the traceback entry it makes
   out of one -- the sequence every Cython-generated module runs when a
   function of its own raises. */
#include "Python.h"
#include "frameobject.h"

/* The C half of the layout `cpyext/frameobject.rs` holds from the other side.
   Each offset is pinned in both, so a field added to one alone stops
   compiling rather than writing somewhere unclaimed. */
_Static_assert(offsetof(PyFrameObject, f_code) == sizeof(PyObject), "f_code moved");
_Static_assert(offsetof(PyFrameObject, f_globals) == sizeof(PyObject) + sizeof(void *),
               "f_globals moved");
_Static_assert(offsetof(PyFrameObject, f_locals) == sizeof(PyObject) + 2 * sizeof(void *),
               "f_locals moved");
_Static_assert(offsetof(PyFrameObject, f_lineno) == sizeof(PyObject) + 3 * sizeof(void *),
               "f_lineno moved");
_Static_assert(offsetof(PyFrameObject, f_back) == sizeof(PyObject) + 4 * sizeof(void *),
               "f_back moved");
_Static_assert(sizeof(PyFrameObject) == sizeof(PyObject) + 5 * sizeof(void *),
               "the frame block changed size");

/* `__Pyx_AddTraceback` verbatim: a code object for a source location this
   runtime never compiled, a frame over it, the line written into the frame,
   and the pair recorded against the exception being propagated. */
static PyObject *add_traceback(PyObject *self, PyObject *args)
{
    const char *funcname;
    const char *filename;
    int py_line;
    PyCodeObject *py_code;
    PyFrameObject *py_frame;
    if (!PyArg_ParseTuple(args, "ssi", &funcname, &filename, &py_line)) {
        return NULL;
    }
    PyErr_SetString(PyExc_ValueError, "boom");
    py_code = PyCode_NewEmpty(filename, funcname, py_line);
    if (!py_code) {
        return NULL;
    }
    py_frame = PyFrame_New(PyThreadState_Get(), py_code,
                           PyModule_GetDict(self), 0);
    if (!py_frame) {
        Py_DECREF(py_code);
        return NULL;
    }
    py_frame->f_lineno = py_line;
    PyTraceBack_Here(py_frame);
    Py_DECREF(py_code);
    Py_DECREF(py_frame);
    return NULL;
}

/* What `PyTraceBack_Here` answers with nothing being propagated. */
static PyObject *here_without_exception(PyObject *self, PyObject *unused)
{
    PyCodeObject *py_code;
    PyFrameObject *py_frame;
    int reported;
    (void)unused;
    py_code = PyCode_NewEmpty("nowhere.c", "nobody", 1);
    if (!py_code) {
        return NULL;
    }
    py_frame = PyFrame_New(PyThreadState_Get(), py_code,
                           PyModule_GetDict(self), 0);
    if (!py_frame) {
        Py_DECREF(py_code);
        return NULL;
    }
    reported = PyTraceBack_Here(py_frame);
    Py_DECREF(py_code);
    Py_DECREF(py_frame);
    return PyLong_FromLong(reported);
}

/* The fields a frame the caller built carries, read back through the block. */
static PyObject *describe_new_frame(PyObject *self, PyObject *args)
{
    const char *filename;
    int py_line;
    PyCodeObject *py_code;
    PyFrameObject *py_frame;
    PyObject *globals;
    PyObject *report;
    if (!PyArg_ParseTuple(args, "si", &filename, &py_line)) {
        return NULL;
    }
    globals = PyModule_GetDict(self);
    py_code = PyCode_NewEmpty(filename, "made", py_line);
    if (!py_code) {
        return NULL;
    }
    py_frame = PyFrame_New(PyThreadState_Get(), py_code, globals, 0);
    if (!py_frame) {
        Py_DECREF(py_code);
        return NULL;
    }
    py_frame->f_lineno = py_line;
    report = Py_BuildValue("OiOO",
                           (PyObject *)py_frame->f_code == (PyObject *)py_code
                               ? Py_True : Py_False,
                           py_frame->f_lineno,
                           py_frame->f_globals ? py_frame->f_globals : Py_None,
                           py_frame->f_locals ? py_frame->f_locals : Py_None);
    Py_DECREF(py_code);
    Py_DECREF(py_frame);
    return report;
}

/* Whether a frame the caller built is what `PyFrame_Check` recognises, and
   what the runtime hands back for its type. */
static PyObject *check_new_frame(PyObject *self, PyObject *unused)
{
    PyCodeObject *py_code;
    PyFrameObject *py_frame;
    PyObject *report;
    (void)unused;
    py_code = PyCode_NewEmpty("checked.c", "checked", 7);
    if (!py_code) {
        return NULL;
    }
    py_frame = PyFrame_New(PyThreadState_Get(), py_code,
                           PyModule_GetDict(self), 0);
    if (!py_frame) {
        Py_DECREF(py_code);
        return NULL;
    }
    report = Py_BuildValue("iO", PyFrame_Check(py_frame),
                           (PyObject *)Py_TYPE(py_frame));
    Py_DECREF(py_code);
    Py_DECREF(py_frame);
    return report;
}

/* A frame this runtime is executing, read through the same fields.  The
   mirror is filled from the interpreter side rather than by the caller, which
   is the other direction the layout is used in. */
static PyObject *describe_running_frame(PyObject *self, PyObject *arg)
{
    PyFrameObject *py_frame;
    (void)self;
    if (!PyFrame_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "not a frame");
        return NULL;
    }
    py_frame = (PyFrameObject *)arg;
    return Py_BuildValue("OiOO",
                         py_frame->f_code ? (PyObject *)py_frame->f_code : Py_None,
                         py_frame->f_lineno,
                         py_frame->f_globals ? py_frame->f_globals : Py_None,
                         py_frame->f_back ? (PyObject *)py_frame->f_back : Py_None);
}

static PyMethodDef methods[] = {
    {"describe_running_frame", describe_running_frame, METH_O, NULL},
    {"add_traceback", add_traceback, METH_VARARGS, NULL},
    {"here_without_exception", here_without_exception, METH_NOARGS, NULL},
    {"describe_new_frame", describe_new_frame, METH_VARARGS, NULL},
    {"check_new_frame", check_new_frame, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_frames", NULL,
                                 -1, methods};

PyMODINIT_FUNC PyInit_cpyext_frames(void)
{
    return PyModule_Create(&def);
}
