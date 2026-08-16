# pyre-check: gate=1
C = type('C', (), {1: 2})
result = C.__dict__[1] == 2

assert result
