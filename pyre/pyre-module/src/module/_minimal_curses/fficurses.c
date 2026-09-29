/* fficurses.py separate_module_sources.
 * term.h macros rename ordinary identifiers, so the wrappers stay in
 * this translation unit.
 */
#if defined(PYRE_CURSES_INCLUDE_NCURSES_PREFIX)
#include <ncurses/curses.h>
#include <ncurses/term.h>
#else
#include <curses.h>
#include <term.h>
#endif

int rpy_curses_setupterm(char *t, int fd, int *errret) {
    return setupterm(t, fd, errret);
}

char *rpy_curses_tigetstr(char *capname) {
    char *res = tigetstr(capname);
    if (res == (char *)-1)
        res = NULL;
    return res;
}

char *rpy_curses_tparm(char *str, int x0, int x1, int x2, int x3,
                       int x4, int x5, int x6, int x7, int x8) {
    return tparm(str, x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
