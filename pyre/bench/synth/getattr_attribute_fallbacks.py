# No `max-pypy-ratio`: this fixture has no loop, and its jitstats record
# `loops_compiled=0` on every backend -- nothing is compiled, so what a
# ratio would compare is two interpreters' startup. It reads between 1.0x
# and 12.2x across the runners with no change in what it exercises.
# descroperation.py _handle_getattribute / objspace.py getattr fast path:
# an AttributeError raised by a custom __getattribute__ OR by a descriptor
# __get__ falls back to __getattr__ rather than propagating.


class CustomGetattribute:
    def __getattribute__(self, name):
        if name == 'boom':
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        return 'ga:' + name


class RaisingDescr:
    def __get__(self, obj, objtype=None):
        raise AttributeError('descr')


class WithDescr:
    x = RaisingDescr()

    def __getattr__(self, name):
        return 'wd:' + name


class WithDescrNoHook:
    x = RaisingDescr()


def main():
    c = CustomGetattribute()
    c.ok = 1
    # custom __getattribute__ AttributeError -> __getattr__
    print('getattribute_fallback', c.boom)
    # non-error access still works through the custom slot
    print('getattribute_ok', c.ok)

    # descriptor __get__ AttributeError -> __getattr__
    print('descr_fallback', WithDescr().x)

    # no __getattr__ -> the AttributeError propagates
    try:
        WithDescrNoHook().x
        print('descr_no_hook', 'BUG')
    except AttributeError:
        print('descr_no_hook', 'AttributeError')


main()
