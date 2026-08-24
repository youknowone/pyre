//! The `datetime` C API: the table an extension binds, the constructors it
//! reaches through that table, the accessors it spells as macros, and the two
//! words it reads straight out of a block.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import datetime
import gc

import cpyext_datetime as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# ── the table ──────────────────────────────────────────────────────────

table = m.table()
eq('the table names date', table['date'], datetime.date)
eq('the table names datetime', table['datetime'], datetime.datetime)
eq('the table names time', table['time'], datetime.time)
eq('the table names timedelta', table['timedelta'], datetime.timedelta)
eq('the table names tzinfo', table['tzinfo'], datetime.tzinfo)
eq('the table carries the UTC singleton',
   table['utc'] is datetime.timezone.utc, True)

# A word an extension reads out of a block exists only if the block was
# allocated with room for it, and `date` is the one class here with no such
# word.
sizes = m.sizes()
eq('a time block is shaped as a datetime one', sizes['time'], sizes['datetime'])
assert sizes['datetime'] > sizes['date'], sizes
assert sizes['timedelta'] > sizes['date'], sizes


# ── the constructors ───────────────────────────────────────────────────

eq('a date built from C', m.make_date(2021, 6, 7), datetime.date(2021, 6, 7))
eq('a datetime built from C',
   m.make_datetime(2021, 6, 7, 8, 9, 10, 11),
   datetime.datetime(2021, 6, 7, 8, 9, 10, 11))
eq('a time built from C',
   m.make_time(8, 9, 10, 11), datetime.time(8, 9, 10, 11))
eq('a timedelta built from C',
   m.make_delta(1, 2, 3),
   datetime.timedelta(days=1, seconds=2, microseconds=3))

# The class the macros hand the constructor is the table's slot, so what comes
# back is that class and not a mirror of it.
eq('the class a constructed date is of',
   type(m.make_date(2021, 6, 7)) is datetime.date, True)
eq('the class a constructed datetime is of',
   type(m.make_datetime(2021, 6, 7, 8, 9, 10, 11)) is datetime.datetime, True)

# `fold` is the one keyword, so the constructor has to reach it past seven
# positional arguments.
folded = m.make_datetime_fold(2021, 6, 7, 8, 9, 10, 11, 1)
eq('the fold of a datetime built from C', folded.fold, 1)
eq('and the rest of that datetime',
   folded.replace(fold=0), datetime.datetime(2021, 6, 7, 8, 9, 10, 11))
eq('the fold of a time built from C', m.make_time_fold(8, 9, 10, 11, 1).fold, 1)
eq('a time built with no fold', m.make_time_fold(8, 9, 10, 11, 0).fold, 0)


# ── time zones ─────────────────────────────────────────────────────────

aware = m.make_aware(90, 'plus-ninety')
eq('the zone of a datetime built from C',
   aware.tzinfo, datetime.timezone(datetime.timedelta(minutes=90), 'plus-ninety'))
eq('and the offset it answers', aware.utcoffset(), datetime.timedelta(minutes=90))
eq('and the name it answers', aware.tzname(), 'plus-ninety')
eq('the rest of it', aware.replace(tzinfo=None),
   datetime.datetime(2021, 6, 7, 8, 9, 10, 11))

eq('a zone built with no name',
   m.make_aware(-60, '').utcoffset(), datetime.timedelta(minutes=-60))
eq('a zero offset with no name is the UTC singleton',
   m.make_aware(0, '').tzinfo is datetime.timezone.utc, True)


# ── the timestamp constructors ─────────────────────────────────────────

stamp = 1600000000.5
as_datetime, as_date = m.from_timestamp(stamp)
eq('a datetime from a timestamp',
   as_datetime, datetime.datetime.fromtimestamp(stamp))
eq('a date from a timestamp', as_date, datetime.date.fromtimestamp(stamp))


# ── the accessors ──────────────────────────────────────────────────────

fields = m.fields_of(aware)
eq('the fields of a datetime',
   (fields['year'], fields['month'], fields['day']), (2021, 6, 7))
eq('the time fields of a datetime',
   (fields['hour'], fields['minute'], fields['second'], fields['microsecond']),
   (8, 9, 10, 11))
eq('the fold of a datetime', fields['fold'], 0)
eq('the zone an accessor reads', fields['tzinfo'] is aware.tzinfo, True)

naive = datetime.datetime(2021, 6, 7, 8, 9, 10, 11)
eq('a naive datetime has no zone to read',
   m.fields_of(naive)['tzinfo'] is None, True)
eq('the fold an accessor reads', m.fields_of(folded)['fold'], 1)

# A `date` answers year, month and day; the rest name attributes it does not
# have, and reading one is not an error.
day = m.fields_of(datetime.date(2021, 6, 7))
eq('the fields of a date', (day['year'], day['month'], day['day']), (2021, 6, 7))
eq('the hour of a date', day['hour'], 0)
eq('the zone of a date', day['tzinfo'] is None, True)

zoned_time = datetime.time(8, 9, 10, 11, tzinfo=datetime.timezone.utc)
times = m.time_fields_of(zoned_time)
eq('the fields of a time',
   (times['hour'], times['minute'], times['second'], times['microsecond']),
   (8, 9, 10, 11))
eq('the fold of a time', times['fold'], 0)
eq('the zone of a time', times['tzinfo'] is datetime.timezone.utc, True)

eq('the fields of a timedelta',
   m.delta_fields_of(datetime.timedelta(days=1, seconds=2, microseconds=3)),
   {'days': 1, 'seconds': 2, 'microseconds': 3})
eq('the fields of a negative timedelta',
   m.delta_fields_of(datetime.timedelta(microseconds=-1)),
   {'days': -1, 'seconds': 86399, 'microseconds': 999999})


# ── the words a block carries ──────────────────────────────────────────

# `hastzinfo` decides whether the word beside it is a reference at all, so a
# block that answered 1 with nothing behind it would be read as one.
eq('a naive datetime block', m.block_of(naive), (0, None))
eq('an aware datetime block', m.block_of(aware), (1, aware.tzinfo))
eq('a naive time block', m.block_of(datetime.time(1, 2)), (0, None))
eq('an aware time block',
   m.block_of(zoned_time), (1, datetime.timezone.utc))
eq('a timedelta block',
   m.block_of(datetime.timedelta(days=1, seconds=2, microseconds=3)), (1, 2, 3))


# ── the check functions ────────────────────────────────────────────────

def answers(object):
    return {name for name, held in m.checks_of(object).items() if held}


eq('a date', answers(datetime.date(2021, 6, 7)), {'date', 'date_exact'})
eq('a datetime is a date too',
   answers(naive), {'date', 'datetime', 'datetime_exact'})
eq('a time', answers(datetime.time(1, 2)), {'time', 'time_exact'})
eq('a timedelta', answers(datetime.timedelta(1)), {'delta', 'delta_exact'})
eq('a timezone is a tzinfo', answers(datetime.timezone.utc), {'tzinfo'})
eq('an int is none of them', answers(3), set())


# ── classes derived from these in Python ───────────────────────────────

class Day(datetime.date):
    pass


class Moment(datetime.datetime):
    pass


eq('a subclass of date', answers(Day(2021, 6, 7)), {'date'})
eq('a subclass of datetime', answers(Moment(2021, 6, 7)), {'date', 'datetime'})

# The word is filled on whatever block the size test made room for, and a
# subclass of `datetime` is sized as its base.
moment = Moment(2021, 6, 7, 8, 9, 10, 11, tzinfo=datetime.timezone.utc)
eq('the block of an instance of a subclass',
   m.block_of(moment), (1, datetime.timezone.utc))
eq('and its accessors', m.fields_of(moment)['tzinfo'] is datetime.timezone.utc, True)


# ── the reference a block holds on its zone ────────────────────────────

for index in range(200):
    m.block_of(m.make_aware(index % 60, ''))
gc.collect()
eq('the module still builds after those blocks went',
   m.make_date(2000, 1, 1), datetime.date(2000, 1, 1))
eq('and the zone of one that is still held', m.block_of(aware), (1, aware.tzinfo))

print('cpyext-datetime-ok')
"#;

#[test]
fn drives_the_datetime_c_api() {
    let fixtures = Fixtures::new("cpyext-datetime");
    fixtures.compile("cpyext_datetime");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-datetime-ok");
}
