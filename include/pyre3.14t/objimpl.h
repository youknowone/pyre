/* The object allocator and the collector's no-ops.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_OBJIMPL_H
#define PYRE_OBJIMPL_H

#ifdef __cplusplus
extern "C" {
#endif
#define PyObject_New(type, tp) ((type *)PyType_GenericAlloc((tp), 0))
#define PyObject_GC_New(type, tp) PyObject_New(type, tp)

/* The object allocator.  `PyObject_Free` is the deallocator for all three, and
   for a block `tp_alloc` handed out; the `PyMem_*` family above is a different
   allocator and its blocks must go back to `PyMem_Free`. */

#define PyObject_GC_Del(ob) PyObject_Del(ob)

/* The body of a `tp_traverse`: report one field, and stop at the first
   visitor that answers non-zero.  `visit` and `arg` are the names
   `traverseproc` gives its own parameters, which is what lets the macro read
   as a single argument at the call site. */
#define Py_VISIT(op)                                            \
    do {                                                        \
        if (op) {                                               \
            int vret = visit((PyObject *)(op), arg);            \
            if (vret)                                           \
                return vret;                                    \
        }                                                       \
    } while (0)

/* The reference header reads the flag out of the collector's own header word
   in front of the block; here the answer is a call. */
#define _PyGC_FINALIZED(o) PyObject_GC_IsFinalized((PyObject *)(o))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_OBJIMPL_H */
