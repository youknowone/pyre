# Missing-attribute lookup on a young instance. Materialises the
# AttributeError through `object_getattr_miss` and
# `attribute_error_with_context`, so the receiver and name string are
# live across the collecting `w_str_new_managed` / `w_list_new` path.
# Expected: AttributeError absent
class C:
    pass


try:
    C().absent
except AttributeError as e:
    print(type(e).__name__, e.name)
