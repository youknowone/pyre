/* `dict`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_DICTOBJECT_H
#define PYRE_DICTOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* dict. */

#define PyDoc_STRVAR(name, str) static const char name[] = str
#define PyDoc_STR(str) str

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_DICTOBJECT_H */
