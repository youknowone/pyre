# pyre-check: no-cpython
import gc


class Render:
    def __init__(self, result):
        self.result = result

    def __str__(self):
        return self.result


runtime_text = "runtime-exception-value-" + str(id(gc))
rendered_text = "rendered-exception-value-" + str(id(runtime_text))

empty = ValueError()
single_text = ValueError(runtime_text)
single_object = ValueError(Render(rendered_text))
multiple = ValueError(runtime_text, 1)
special = KeyError(runtime_text)

empty_direct = BaseException.__str__(empty)
empty_ordinary = str(empty)
text_direct = BaseException.__str__(single_text)
text_ordinary = str(single_text)
object_direct = BaseException.__str__(single_object)
object_ordinary = str(single_object)
multiple_direct = BaseException.__str__(multiple)
multiple_ordinary = str(multiple)
special_direct = KeyError.__str__(special)
special_ordinary = str(special)

assert empty_direct is empty_ordinary
assert text_direct is text_ordinary is runtime_text
assert object_direct is object_ordinary is rendered_text
assert multiple_direct == multiple_ordinary
assert special_direct == special_ordinary == repr(runtime_text)

# The empty result is the translated `space.newtext('')` prebuilt.  The
# dynamic results below must instead be ordinary, collectable GC objects.
for result in (
    text_direct,
    text_ordinary,
    object_direct,
    object_ordinary,
    multiple_direct,
    multiple_ordinary,
    special_direct,
    special_ordinary,
):
    assert any(obj is result for obj in gc.get_objects())

print("exception str results preserve identity and are collectable")
