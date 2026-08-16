/* `PyMethodDef` and the calling conventions it selects.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_METHODOBJECT_H
#define PYRE_METHODOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
struct PyMethodDef {
    const char *ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    const char *ml_doc;
};

#define METH_VARARGS 0x0001
#define METH_KEYWORDS 0x0002
#define METH_NOARGS 0x0004
#define METH_O 0x0008
#define METH_CLASS 0x0010
#define METH_STATIC 0x0020
#define METH_COEXIST 0x0040
#define METH_FASTCALL 0x0080

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_METHODOBJECT_H */
