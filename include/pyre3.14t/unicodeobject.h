/* `str`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_UNICODEOBJECT_H
#define PYRE_UNICODEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* str. */

/* A `str` here is a mirror rather than a compact object with a readable
   `length` field, so the fast spelling is the call. */
#define PyUnicode_GET_LENGTH(op) PyUnicode_GetLength((PyObject *)(op))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_UNICODEOBJECT_H */
