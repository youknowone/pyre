/* The `datetime` C API: the table `PyDateTime_IMPORT` binds, the constructors
   reached through it, and the accessors an extension spells as macros.

   The shape is Cython's `cpython.datetime`, which every extension that packs a
   `datetime` goes through -- msgpack among them. */
#include "Python.h"
#include "datetime.h"

/* Whether the table was bound, and whether each of its type slots is the class
   of the same name.  Everything below reaches through it, so a failure here is
   the first thing to look at. */
static int table_bound(void)
{
    if (PyDateTimeAPI != NULL) {
        return 1;
    }
    PyErr_SetString(PyExc_AssertionError, "PyDateTimeAPI is NULL");
    return 0;
}

static PyObject *table(PyObject *self, PyObject *module)
{
    (void)self;
    (void)module;
    if (!table_bound()) {
        return NULL;
    }
    return Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O}",
        "date", (PyObject *)PyDateTimeAPI->DateType,
        "datetime", (PyObject *)PyDateTimeAPI->DateTimeType,
        "time", (PyObject *)PyDateTimeAPI->TimeType,
        "timedelta", (PyObject *)PyDateTimeAPI->DeltaType,
        "tzinfo", (PyObject *)PyDateTimeAPI->TZInfoType,
        "utc", PyDateTime_TimeZone_UTC);
}

/* `tp_basicsize` of each of the table's types.  A word an extension reads
   straight out of a block exists only if the block was allocated with room
   for it, and that room is this number. */
static PyObject *sizes(PyObject *self, PyObject *module)
{
    (void)self;
    (void)module;
    if (!table_bound()) {
        return NULL;
    }
    return Py_BuildValue(
        "{s:n,s:n,s:n,s:n}",
        "date", PyDateTimeAPI->DateType->tp_basicsize,
        "datetime", PyDateTimeAPI->DateTimeType->tp_basicsize,
        "time", PyDateTimeAPI->TimeType->tp_basicsize,
        "timedelta", PyDateTimeAPI->DeltaType->tp_basicsize);
}

static PyObject *make_date(PyObject *self, PyObject *args)
{
    int year, month, day;
    (void)self;
    if (!PyArg_ParseTuple(args, "iii", &year, &month, &day)) {
        return NULL;
    }
    return PyDate_FromDate(year, month, day);
}

static PyObject *make_datetime(PyObject *self, PyObject *args)
{
    int year, month, day, hour, minute, second, usecond;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiiiiii", &year, &month, &day, &hour, &minute,
                          &second, &usecond)) {
        return NULL;
    }
    return PyDateTime_FromDateAndTime(year, month, day, hour, minute, second,
                                      usecond);
}

static PyObject *make_datetime_fold(PyObject *self, PyObject *args)
{
    int year, month, day, hour, minute, second, usecond, fold;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiiiiiii", &year, &month, &day, &hour, &minute,
                          &second, &usecond, &fold)) {
        return NULL;
    }
    return PyDateTime_FromDateAndTimeAndFold(year, month, day, hour, minute,
                                             second, usecond, fold);
}

static PyObject *make_time(PyObject *self, PyObject *args)
{
    int hour, minute, second, usecond;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiii", &hour, &minute, &second, &usecond)) {
        return NULL;
    }
    return PyTime_FromTime(hour, minute, second, usecond);
}

static PyObject *make_time_fold(PyObject *self, PyObject *args)
{
    int hour, minute, second, usecond, fold;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiiii", &hour, &minute, &second, &usecond,
                          &fold)) {
        return NULL;
    }
    return PyTime_FromTimeAndFold(hour, minute, second, usecond, fold);
}

static PyObject *make_delta(PyObject *self, PyObject *args)
{
    int days, seconds, useconds;
    (void)self;
    if (!PyArg_ParseTuple(args, "iii", &days, &seconds, &useconds)) {
        return NULL;
    }
    return PyDelta_FromDSU(days, seconds, useconds);
}

/* An aware `datetime` built entirely from C: the offset goes through
   `PyDelta_FromDSU`, the zone through `PyTimeZone_FromOffsetAndName`, and the
   result through the constructor the table carries. */
static PyObject *make_aware(PyObject *self, PyObject *args)
{
    int minutes;
    PyObject *offset;
    PyObject *zone;
    PyObject *made;
    const char *name;
    (void)self;
    if (!PyArg_ParseTuple(args, "is", &minutes, &name)) {
        return NULL;
    }
    offset = PyDelta_FromDSU(0, minutes * 60, 0);
    if (offset == NULL) {
        return NULL;
    }
    if (name[0] == '\0') {
        zone = PyTimeZone_FromOffset(offset);
    } else {
        PyObject *label = PyUnicode_FromString(name);
        if (label == NULL) {
            Py_DECREF(offset);
            return NULL;
        }
        zone = PyTimeZone_FromOffsetAndName(offset, label);
        Py_DECREF(label);
    }
    Py_DECREF(offset);
    if (zone == NULL) {
        return NULL;
    }
    made = PyDateTimeAPI->DateTime_FromDateAndTime(2021, 6, 7, 8, 9, 10, 11,
                                                   zone,
                                                   PyDateTimeAPI->DateTimeType);
    Py_DECREF(zone);
    return made;
}

static PyObject *from_timestamp(PyObject *self, PyObject *args)
{
    PyObject *stamp;
    PyObject *forwarded;
    PyObject *pair;
    PyObject *as_datetime;
    PyObject *as_date;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &stamp)) {
        return NULL;
    }
    forwarded = PyTuple_Pack(1, stamp);
    if (forwarded == NULL) {
        return NULL;
    }
    as_datetime = PyDateTimeAPI->DateTime_FromTimestamp(
        (PyObject *)PyDateTimeAPI->DateTimeType, forwarded, NULL);
    if (as_datetime == NULL) {
        Py_DECREF(forwarded);
        return NULL;
    }
    as_date = PyDateTimeAPI->Date_FromTimestamp(
        (PyObject *)PyDateTimeAPI->DateType, forwarded);
    Py_DECREF(forwarded);
    if (as_date == NULL) {
        Py_DECREF(as_datetime);
        return NULL;
    }
    pair = Py_BuildValue("OO", as_datetime, as_date);
    Py_DECREF(as_datetime);
    Py_DECREF(as_date);
    return pair;
}

/* Every accessor, read off one object. */
static PyObject *fields_of(PyObject *self, PyObject *object)
{
    (void)self;
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:O}",
        "year", PyDateTime_GET_YEAR(object),
        "month", PyDateTime_GET_MONTH(object),
        "day", PyDateTime_GET_DAY(object),
        "fold", PyDateTime_GET_FOLD(object),
        "hour", PyDateTime_DATE_GET_HOUR(object),
        "minute", PyDateTime_DATE_GET_MINUTE(object),
        "second", PyDateTime_DATE_GET_SECOND(object),
        "microsecond", PyDateTime_DATE_GET_MICROSECOND(object),
        "tzinfo", PyDateTime_DATE_GET_TZINFO(object));
}

static PyObject *time_fields_of(PyObject *self, PyObject *object)
{
    (void)self;
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:O}",
        "hour", PyDateTime_TIME_GET_HOUR(object),
        "minute", PyDateTime_TIME_GET_MINUTE(object),
        "second", PyDateTime_TIME_GET_SECOND(object),
        "microsecond", PyDateTime_TIME_GET_MICROSECOND(object),
        "fold", PyDateTime_TIME_GET_FOLD(object),
        "tzinfo", PyDateTime_TIME_GET_TZINFO(object));
}

static PyObject *delta_fields_of(PyObject *self, PyObject *object)
{
    (void)self;
    return Py_BuildValue("{s:i,s:i,s:i}",
                         "days", PyDateTime_DELTA_GET_DAYS(object),
                         "seconds", PyDateTime_DELTA_GET_SECONDS(object),
                         "microseconds",
                         PyDateTime_DELTA_GET_MICROSECONDS(object));
}

/* The two words an extension reads through the block rather than through a
   call.  `hastzinfo` decides whether `tzinfo` is a reference at all, so a
   block that answered 1 with nothing behind it would be read as one. */
static PyObject *block_of(PyObject *self, PyObject *object)
{
    (void)self;
    if (PyDateTime_Check(object)) {
        PyDateTime_DateTime *block = (PyDateTime_DateTime *)object;
        return Py_BuildValue("iO", (int)block->hastzinfo,
                             block->tzinfo == NULL ? Py_None : block->tzinfo);
    }
    if (PyTime_Check(object)) {
        PyDateTime_Time *block = (PyDateTime_Time *)object;
        return Py_BuildValue("iO", (int)block->hastzinfo,
                             block->tzinfo == NULL ? Py_None : block->tzinfo);
    }
    if (PyDelta_Check(object)) {
        PyDateTime_Delta *block = (PyDateTime_Delta *)object;
        return Py_BuildValue("iii", block->days, block->seconds,
                             block->microseconds);
    }
    PyErr_SetString(PyExc_TypeError, "not a datetime, time or timedelta");
    return NULL;
}

static PyObject *checks_of(PyObject *self, PyObject *object)
{
    (void)self;
    return Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
        "date", PyDate_Check(object) ? Py_True : Py_False,
        "date_exact", PyDate_CheckExact(object) ? Py_True : Py_False,
        "datetime", PyDateTime_Check(object) ? Py_True : Py_False,
        "datetime_exact", PyDateTime_CheckExact(object) ? Py_True : Py_False,
        "time", PyTime_Check(object) ? Py_True : Py_False,
        "time_exact", PyTime_CheckExact(object) ? Py_True : Py_False,
        "delta", PyDelta_Check(object) ? Py_True : Py_False,
        "delta_exact", PyDelta_CheckExact(object) ? Py_True : Py_False,
        "tzinfo", PyTZInfo_Check(object) ? Py_True : Py_False,
        "tzinfo_exact", PyTZInfo_CheckExact(object) ? Py_True : Py_False);
}

static PyMethodDef methods[] = {
    {"table", table, METH_NOARGS, NULL},
    {"sizes", sizes, METH_NOARGS, NULL},
    {"make_date", make_date, METH_VARARGS, NULL},
    {"make_datetime", make_datetime, METH_VARARGS, NULL},
    {"make_datetime_fold", make_datetime_fold, METH_VARARGS, NULL},
    {"make_time", make_time, METH_VARARGS, NULL},
    {"make_time_fold", make_time_fold, METH_VARARGS, NULL},
    {"make_delta", make_delta, METH_VARARGS, NULL},
    {"make_aware", make_aware, METH_VARARGS, NULL},
    {"from_timestamp", from_timestamp, METH_VARARGS, NULL},
    {"fields_of", fields_of, METH_O, NULL},
    {"time_fields_of", time_fields_of, METH_O, NULL},
    {"delta_fields_of", delta_fields_of, METH_O, NULL},
    {"block_of", block_of, METH_O, NULL},
    {"checks_of", checks_of, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static int datetime_exec(PyObject *module)
{
    (void)module;
    PyDateTime_IMPORT;
    if (PyDateTimeAPI == NULL) {
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)datetime_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_datetime",
    "pyre cpyext datetime module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_datetime(void)
{
    return PyModuleDef_Init(&moduledef);
}
