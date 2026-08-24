/* The variadic entry points: argument parsing and value building.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The non-variadic entry points these are built out of are
 * declared together in `pyre_decl.h`, which is generated.
 *
 * `PyArg_ParseTuple` and `Py_BuildValue` are C functions with C bodies, and
 * pyre compiles them into the interpreter and exports them from it, which is
 * where `pypy/module/cpyext/src/getargs.c` and `.../modsupport.c` put the
 * same ones. The bodies live beside their peers in
 * `pyre/pyre-interpreter/src/cpyext/src/`; only the declarations are here.
 */
#ifndef PYRE_MODSUPPORT_H
#define PYRE_MODSUPPORT_H

#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(int) PyArg_Parse(PyObject *, const char *, ...);
PyAPI_FUNC(int) PyArg_ParseTuple(PyObject *, const char *, ...);
PyAPI_FUNC(int) PyArg_ParseTupleAndKeywords(PyObject *, PyObject *, const char *,
                                            PY_CXX_CONST char *const *, ...);
PyAPI_FUNC(int) PyArg_UnpackTuple(PyObject *, const char *, Py_ssize_t, Py_ssize_t, ...);
PyAPI_FUNC(int) _PyArg_CheckPositional(const char *, Py_ssize_t, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(PyObject *) Py_BuildValue(const char *, ...);
PyAPI_FUNC(PyObject *) Py_VaBuildValue(const char *, va_list);

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_MODSUPPORT_H */
