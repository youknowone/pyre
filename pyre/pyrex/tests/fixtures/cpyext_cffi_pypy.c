/* Minimal generated-PyPy-mode CFFI module.  The layout and startup exchange
   are emitted by cffi.recompiler._make_c_source; keeping this fixture free of
   Python.h also proves that the interpreter-level CFFI path does not fall
   through to PyInit_* or cpyext. */

#include <stddef.h>
#include <stdint.h>

typedef intptr_t cffi_opcode_t;

struct cffi_type_context {
    cffi_opcode_t *types;
    const void *globals;
    const void *fields;
    const void *struct_unions;
    const void *enums;
    const void *typenames;
    int num_globals;
    int num_struct_unions;
    int num_enums;
    int num_typenames;
    const char *const *includes;
    int num_types;
    int flags;
};

static struct cffi_type_context context = {
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, NULL, 0, 0,
};

__attribute__((visibility("default"))) void
_cffi_pypyinit_cpyext_cffi_pypy(const void *p[])
{
    p[0] = (const void *)0x2601;
    p[1] = &context;
}
