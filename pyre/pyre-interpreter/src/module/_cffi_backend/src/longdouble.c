/* `long double` accessors.
 *
 * Rust has no `long double`, and its width and representation differ per
 * target (80-bit x87 padded to 16 bytes on x86-64 System V, plain `double`
 * on aarch64 Darwin, 128-bit quad elsewhere).  The C compiler is the only
 * thing that knows, so every operation on that representation is behind one
 * of these shims.
 *
 * `misc.py` reaches for C the same way: `pypy__is_nonnull_longdouble` is an
 * `ExternalCompilationInfo` snippet, and `longdouble2str` calls `sprintf`
 * with `%LE`.
 */

#include "src/precommondefs.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

struct pyre_cffi_align_long_double {
    char x;
    long double y;
};

RPY_EXTERN size_t pyre_cffi_sizeof_long_double(void)
{
    return sizeof(long double);
}

RPY_EXTERN size_t pyre_cffi_alignof_long_double(void)
{
    return offsetof(struct pyre_cffi_align_long_double, y);
}

RPY_EXTERN double pyre_cffi_read_long_double(const char *p)
{
    long double value;
    memcpy(&value, p, sizeof(long double));
    return (double)value;
}

RPY_EXTERN void pyre_cffi_write_long_double(char *p, double v)
{
    long double value = (long double)v;
    memcpy(p, &value, sizeof(long double));
}

/* `misc.py:_is_nonnull_longdouble` — the comparison has to happen in the
 * `long double` domain, so it cannot be done on the copied-out `double`. */
RPY_EXTERN int pyre_cffi_nonnull_long_double(const char *p)
{
    long double value;
    memcpy(&value, p, sizeof(long double));
    return value != 0.0L;
}

/* `misc.py:longdouble2str`. */
RPY_EXTERN void pyre_cffi_str_long_double(const char *p, char *out, size_t n)
{
    long double value;
    memcpy(&value, p, sizeof(long double));
    snprintf(out, n, "%LE", value);
}
