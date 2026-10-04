# CPython-suite gap: `test_exceptions` stringifies a constructed
# UnicodeDecodeError and never deletes encoding, object, or reason.
#
# parity-tests reason: UnicodeDecodeError_str returns empty when object
# is NULL. PyObject_Str of a NULL encoding or reason spells `<NULL>`.
# A stored None is present: object None is TypeError from
# check_unicode_error_attribute, encoding/reason None stringify as
# `None`. PyMember_SetOne clears `_Py_T_OBJECT` to NULL.
#
# pyre-check: pypy-diverges: `readwrite_attrproperty_w` installs no `fdel`,
# so `del e.object` raises AttributeError. `W_UnicodeDecodeError.descr_str`
# returns empty when object is None.
def decode_deleted_object_is_empty():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    del exc.object
    assert exc.object is None
    assert str(exc) == "", str(exc)


def decode_assigned_none_object_typeerrors():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    exc.object = None
    try:
        str(exc)
    except TypeError as err:
        assert str(err) == "UnicodeError 'object' attribute must be a bytes", str(err)
    else:
        raise AssertionError("expected TypeError")


def decode_deleted_encoding_spells_null():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    del exc.encoding
    assert (
        str(exc) == "'<NULL>' codec can't decode byte 0xff in position 0: bad"
    ), str(exc)


def decode_assigned_none_encoding_spells_none():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    exc.encoding = None
    assert (
        str(exc) == "'None' codec can't decode byte 0xff in position 0: bad"
    ), str(exc)


def decode_deleted_reason_spells_null():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    del exc.reason
    assert (
        str(exc) == "'utf-8' codec can't decode byte 0xff in position 0: <NULL>"
    ), str(exc)


def decode_assigned_none_reason_spells_none():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    exc.reason = None
    assert (
        str(exc) == "'utf-8' codec can't decode byte 0xff in position 0: None"
    ), str(exc)


def encode_deleted_object_is_empty():
    exc = UnicodeEncodeError("utf-8", "é", 0, 1, "bad")
    del exc.object
    assert str(exc) == "", str(exc)


def encode_assigned_none_object_typeerrors():
    exc = UnicodeEncodeError("utf-8", "é", 0, 1, "bad")
    exc.object = None
    try:
        str(exc)
    except TypeError as err:
        assert str(err) == "UnicodeError 'object' attribute must be a string", str(err)
    else:
        raise AssertionError("expected TypeError")


def translate_deleted_reason_spells_null():
    exc = UnicodeTranslateError("é", 0, 1, "bad")
    del exc.reason
    assert (
        str(exc) == "can't translate character '\\xe9' in position 0: <NULL>"
    ), str(exc)


def translate_assigned_none_reason_spells_none():
    exc = UnicodeTranslateError("é", 0, 1, "bad")
    exc.reason = None
    assert str(exc) == "can't translate character '\\xe9' in position 0: None", str(exc)


decode_deleted_object_is_empty()
decode_assigned_none_object_typeerrors()
decode_deleted_encoding_spells_null()
decode_assigned_none_encoding_spells_none()
decode_deleted_reason_spells_null()
decode_assigned_none_reason_spells_none()
encode_deleted_object_is_empty()
encode_assigned_none_object_typeerrors()
translate_deleted_reason_spells_null()
translate_assigned_none_reason_spells_none()
print("OK")
