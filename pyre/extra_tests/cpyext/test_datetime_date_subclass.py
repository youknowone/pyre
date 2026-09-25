# cpyext-fixture: cpyext_datetime
# cpyext-expect: date-subclass-layout-ok

# A pure `date` subclass follows the shrunk date header. The tzinfo words
# belong to `datetime` and `time`, not to a class that only inherits `date`.

import datetime
import cpyext_datetime as m

class D(datetime.date):
    pass

class DT(datetime.datetime):
    pass

class T(datetime.time):
    pass

date_size = m.basicsize_of(datetime.date)
assert date_size > 0, date_size
assert m.basicsize_of(D) == date_size, (m.basicsize_of(D), date_size)
assert m.type_data_size(D) == 0, m.type_data_size(D)
assert m.basicsize_of(DT) == m.basicsize_of(datetime.datetime)
assert m.type_data_size(DT) == 0, m.type_data_size(DT)
assert m.basicsize_of(T) == m.basicsize_of(datetime.time)
assert m.type_data_size(T) == 0, m.type_data_size(T)
print('date-subclass-layout-ok')
