/* `list`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_LISTOBJECT_H
#define PYRE_LISTOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* list. */

#define PyList_GET_SIZE(ob) PyList_Size((PyObject *)(ob))
#define PyList_GET_ITEM(ob, i) PyList_GetItem((PyObject *)(ob), (i))
#define PyList_SET_ITEM(ob, i, v) ((void)PyList_SetItem((PyObject *)(ob), (i), (v)))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_LISTOBJECT_H */
