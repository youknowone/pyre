# CPython-suite gap: `test_exceptions` and `test_call` reject a single
# unknown ImportError keyword (`invalid` / `blech`) and never pass more
# keywords than the ParseTuple kwlist length.
#
# parity-tests reason: `vgetargskeywords` reports
# `takes at most N keyword argument(s) (M given)` when
# `nargs + nkwargs > len`, and only then scans for an unknown key.
# `NameError_init` / `AttributeError_init` / `ImportError_init` parse
# against an empty positional tuple, so `nargs` is 0. PyPy interp2app
# names `.__init__` and counts unexpected keys instead.
#
# pyre-check: pypy-diverges: `W_NameError.descr_init` reports
# `NameError.__init__() got an unexpected keyword argument`; two extras
# become `got 2 unexpected keyword arguments`. ImportError uses
# `'bogus' is an invalid keyword argument for ImportError`.
def name_error_two_keywords_overflows():
    try:
        NameError(name="x", bogus=1)
    except TypeError as err:
        assert str(err) == "NameError() takes at most 1 keyword argument (2 given)", err
    else:
        raise AssertionError("expected TypeError")


def name_error_one_unknown_is_unexpected():
    try:
        NameError(bogus=1)
    except TypeError as err:
        assert str(err) == "NameError() got an unexpected keyword argument 'bogus'", err
    else:
        raise AssertionError("expected TypeError")


def unbound_local_error_names_name_error():
    try:
        UnboundLocalError(name="x", bogus=1)
    except TypeError as err:
        assert str(err) == "NameError() takes at most 1 keyword argument (2 given)", err
    else:
        raise AssertionError("expected TypeError")


def attribute_error_three_keywords_overflows():
    try:
        AttributeError(name="n", obj=1, extra=1)
    except TypeError as err:
        assert (
            str(err) == "AttributeError() takes at most 2 keyword arguments (3 given)"
        ), err
    else:
        raise AssertionError("expected TypeError")


def attribute_error_one_unknown_is_unexpected():
    try:
        AttributeError(name="n", bogus=1)
    except TypeError as err:
        assert (
            str(err) == "AttributeError() got an unexpected keyword argument 'bogus'"
        ), err
    else:
        raise AssertionError("expected TypeError")


def import_error_four_keywords_overflows():
    try:
        ImportError(name="n", path="p", name_from="f", extra=1)
    except TypeError as err:
        assert (
            str(err) == "ImportError() takes at most 3 keyword arguments (4 given)"
        ), err
    else:
        raise AssertionError("expected TypeError")


def import_error_one_unknown_is_unexpected():
    try:
        ImportError(name="n", bogus=1)
    except TypeError as err:
        assert str(err) == "ImportError() got an unexpected keyword argument 'bogus'", err
    else:
        raise AssertionError("expected TypeError")


def module_not_found_error_names_import_error():
    try:
        ModuleNotFoundError(name="n", path="p", name_from="f", extra=1)
    except TypeError as err:
        assert (
            str(err) == "ImportError() takes at most 3 keyword arguments (4 given)"
        ), err
    else:
        raise AssertionError("expected TypeError")


name_error_two_keywords_overflows()
name_error_one_unknown_is_unexpected()
unbound_local_error_names_name_error()
attribute_error_three_keywords_overflows()
attribute_error_one_unknown_is_unexpected()
import_error_four_keywords_overflows()
import_error_one_unknown_is_unexpected()
module_not_found_error_names_import_error()
