import configparser
import gc


line = configparser._Line("value ; comment", configparser._CommentSpec(("#",), (";",)))
assert line == "value ; comment"
assert line.clean == "value"
assert line.has_comments is True
assert not hasattr(line, "__dict__")

marker = []
line.clean = marker
gc.collect()
assert line.clean is marker
del line.clean
try:
    line.clean
except AttributeError:
    pass
else:
    raise AssertionError("deleted str-subclass slot remained bound")

parser = configparser.ConfigParser()
parser.read_string("[section]\nanswer = 42 ; comment\n")
assert parser["section"]["answer"] == "42 ; comment"

print("stdlib configparser ok")
