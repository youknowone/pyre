/* The header a module cffi generated compiles against.

   `_cffi_include.h` defines `_CFFI_` on its first line, before it includes
   `Python.h`, and the module it belongs to has two forms: one built out of
   `_cffi_exports` whose init symbol is `PyInit_`, and one exporting
   `_cffi_pypyinit_` and handing its type context to a built-in backend.  It
   picks between them on `PYPY_VERSION`, so this fixture stands where such a
   module stands and reports what it would pick. */

#define _CFFI_
#include <Python.h>

#ifdef PYPY_VERSION
#define INIT_PREFIX "_cffi_pypyinit_"
#define SAW_PYPY_VERSION 1
#else
#define INIT_PREFIX "PyInit_"
#define SAW_PYPY_VERSION 0
#endif

#ifdef PYPY_VERSION_NUM
#define SAW_PYPY_VERSION_NUM 1
#else
#define SAW_PYPY_VERSION_NUM 0
#endif

/* The two macros, the init symbol they choose, and a version macro the marker
   leaves alone. */
static PyObject *chosen(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("(iisi)", SAW_PYPY_VERSION, SAW_PYPY_VERSION_NUM,
                         INIT_PREFIX, PY_VERSION_HEX);
}

static PyMethodDef methods[] = {{"chosen", chosen, METH_NOARGS, NULL},
                                {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_cffi_mode", NULL,
                                 -1, methods};

PyMODINIT_FUNC PyInit_cpyext_cffi_mode(void)
{
    return PyModule_Create(&def);
}
