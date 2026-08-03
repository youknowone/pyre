# pyre-check: max-pypy-ratio=45
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


N = 20000


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

    # Keep this a performance gate rather than an empty-program timing test:
    # exercise every fallback after the loop becomes hot and report a stable
    # aggregate without printing in the measured loop.
    with_descr = WithDescr()
    without_hook = WithDescrNoHook()
    total = 0
    i = 0
    while i < N:
        if c.boom == 'ga:boom':
            total += 1
        if with_descr.x == 'wd:x':
            total += 1
        try:
            without_hook.x
        except AttributeError:
            total += 1
        i += 1
    print('loop', total)


main()
