/* `PyOS_snprintf` and `PyOS_vsnprintf`.
 *
 * Compiled into the interpreter and exported from it, which is where
 * `pypy/module/cpyext/src/mysnprintf.c` puts the same two.  `pyerrors.h`
 * carries the declarations alone.
 */
#include "Python.h"

/* `vsnprintf` with the two guarantees the name adds: at most `size` bytes
   including the terminator, and a NUL at `str[size - 1]` whatever happened.
   The return value is `vsnprintf`'s, so a truncated conversion still reports
   the length it would have needed. */
int PyOS_vsnprintf(char *str, size_t size, const char *format, va_list va)
{
    int written = vsnprintf(str, size, format, va);
    if (size > 0) {
        str[size - 1] = '\0';
    }
    return written;
}

int PyOS_snprintf(char *str, size_t size, const char *format, ...)
{
    va_list va;
    va_start(va, format);
    int written = PyOS_vsnprintf(str, size, format, va);
    va_end(va);
    return written;
}
