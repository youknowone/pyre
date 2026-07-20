import io
import token
import tokenize


source = "x = 0xff + 1\n# comment\nprint(f'{x=}')\n"
tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
assert [item.type for item in tokens[:5]] == [
    token.NAME,
    token.OP,
    token.NUMBER,
    token.OP,
    token.NUMBER,
]
assert tokens[0].start == (1, 0)
assert tokens[0].end == (1, 1)
assert tokens[0].line == "x = 0xff + 1\n"
assert any(item.type == token.COMMENT and item.string == "# comment" for item in tokens)
assert tokens[-1].type == token.ENDMARKER

unicode_tokens = list(tokenize.generate_tokens(io.StringIO("Örter = grün").readline))
assert unicode_tokens[0].string == "Örter"
assert unicode_tokens[0].end == (1, 5)
assert unicode_tokens[2].string == "grün"
assert unicode_tokens[2].end == (1, 12)

lines = iter(['"ЉЊЈЁЂ"'.encode("utf-8")])
encoded = list(
    tokenize._generate_tokens_from_c_tokenizer(
        lines.__next__, encoding="utf-8", extra_tokens=True
    )
)
assert encoded[0] == (token.STRING, '"ЉЊЈЁЂ"', (1, 0), (1, 7), '"ЉЊЈЁЂ"')

try:
    list(tokenize.generate_tokens(lambda: 42))
except TypeError as error:
    assert "non-string" in str(error)
else:
    raise AssertionError("non-string source line accepted")

print("stdlib tokenize ok", len(tokens), len(unicode_tokens))
