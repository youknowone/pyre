/* Frames.
 *
 * An extension includes this by name rather than through `Python.h`, which is
 * where the reference header set puts it too.
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
