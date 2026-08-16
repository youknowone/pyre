/* `tuple`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_TUPLEOBJECT_H
#define PYRE_TUPLEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* tuple. */

#define PyTuple_GET_SIZE(ob) PyTuple_Size((PyObject *)(ob))
#define PyTuple_GET_ITEM(ob, i) PyTuple_GetItem((PyObject *)(ob), (i))
#define PyTuple_SET_ITEM(ob, i, v) ((void)PyTuple_SetItem((PyObject *)(ob), (i), (v)))

/* Variadic, so it is built here out of the non-variadic exports. */
static inline PyObject *PyTuple_Pack(Py_ssize_t n, ...)
{
    PyObject *result = PyTuple_New(n);
    va_list vargs;
    Py_ssize_t i;

    if (result == NULL) {
        return NULL;
    }
    va_start(vargs, n);
    for (i = 0; i < n; i++) {
        PyObject *item = va_arg(vargs, PyObject *);
        Py_XINCREF(item);
        if (PyTuple_SetItem(result, i, item) < 0) {
            va_end(vargs);
            Py_DECREF(result);
            return NULL;
        }
    }
    va_end(vargs);
    return result;
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_TUPLEOBJECT_H */
