/* `slice`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_SLICEOBJECT_H
#define PYRE_SLICEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* slice. */

/* The spelling every extension compiles against.  The exported function is the
   same composition, so a caller reaching either gets the same answer. */
#define PySlice_GetIndicesEx(slice, length, start, stop, step, slicelen) (      \
    PySlice_Unpack((slice), (start), (stop), (step)) < 0 ?                      \
    ((*(slicelen) = 0), -1) :                                                   \
    ((*(slicelen) = PySlice_AdjustIndices((length), (start), (stop), *(step))), \
     0))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_SLICEOBJECT_H */
