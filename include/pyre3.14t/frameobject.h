/* Frames.
 *
 * One of the headers `Python.h` includes -- `PyFrame_Check` is reachable from
 * `Python.h` alone in both reference header sets -- and an extension may also
 * include it by name, which is the spelling most of them use.
 */
#ifndef PYRE_FRAMEOBJECT_H
#define PYRE_FRAMEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* A frame is opaque: nothing outside this runtime reads a field of one, and
   an extension only ever holds a pointer. */
typedef struct _frame PyFrameObject;

#define PyFrame_Check(op) Py_IS_TYPE((op), &PyFrame_Type)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_FRAMEOBJECT_H */
