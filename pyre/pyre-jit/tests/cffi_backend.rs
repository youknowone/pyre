//! End-to-end coverage for the native `_cffi_backend` module, run through the
//! same startup shape as the `pyrex` launcher.
//!
//! The programs are transcribed from `extra_tests/cffi_tests/test_c.py`, whose
//! upstream is `pypy/module/_cffi_backend/test/_backend_test_c.py`.  Only the
//! part of that file the module answers today is covered: primitives, `void`,
//! pointers and arrays.  An uncaught `assert` in a program surfaces here as a
//! failing test, so a passing test means every read-back assertion held.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks, reset_gc_fresh_for_test};

/// The programs share the process-global GC singleton and builtin type state,
/// so they run one at a time regardless of cargo's parallel scheduling.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_harness(program: &str, name: &str) -> Result<(), String> {
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();
    reset_gc_fresh_for_test();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd, cwd.as_os_str());
    importing::add_sys_path_0();
    importing::set_sys_argv(&[std::ffi::OsString::from(name)]);

    let code = compile_source_with_filename(program, Mode::Exec, name)
        .map_err(|e| format!("compile error: {e}"))?;

    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .map_err(|e| format!("frame setup error: {}", e.message))?;

    let canonical = frame.get_w_globals();
    let main_module = pyre_object::w_module_new_aliasing_dict("__main__", canonical);
    importing::set_sys_module("__main__", main_module);

    eval_with_jit(&mut frame, None).map_err(|e| format!("execution error: {}", e.message))?;
    Ok(())
}

fn run_on_worker(program: &'static str, name: &'static str) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_harness(program, name))
        .expect("spawn worker thread");
    handle.join().expect("worker thread panicked").expect(name);
}

/// One program covering every part of `_cffi_backend` the module answers
/// today, transcribed from `extra_tests/cffi_tests/test_c.py`.
///
/// It is a single `#[test]` on purpose: the module memoises every ctype in
/// process-global caches and roots it for the process, so a second program
/// in the same process would meet those caches holding addresses into the
/// heap `reset_gc_fresh_for_test` has already replaced.  Each section runs
/// through `exec` and reports on its own, so one run names every failure
/// rather than only the first.
#[test]
fn the_module_answers_every_primitive_pointer_and_array_case() {
    const PROGRAM: &str = concat!(
        "import sys\n",
        "from _cffi_backend import *\n",
        "from _cffi_backend import _get_common_types, __version__\n",
        "def raises(exc, fn, *a):\n",
        "    try:\n",
        "        fn(*a)\n",
        "    except exc:\n",
        "        return\n",
        "    raise AssertionError('expected ' + exc.__name__)\n",
        "_failures = []\n",
        "def _run(label, body):\n",
        "    try:\n",
        "        exec(body, globals())\n",
        "    except BaseException as e:\n",
        "        _failures.append(label + ': ' + type(e).__name__ + ': ' + str(e))\n",
        "        print('FAILED', _failures[-1])\n",
        "_run('''primitive_types_are_memoised_and_inspectable''', r'''",
        "assert __version__ == \"1.18.0.dev0\", __version__\n",
        "raises(KeyError, new_primitive_type, \"foo\")\n",
        "p = new_primitive_type(\"signed char\")\n",
        "assert repr(p) == \"<ctype 'signed char'>\", repr(p)\n",
        "assert p is new_primitive_type(\"signed char\")\n",
        "assert p.kind == \"primitive\"\n",
        "assert p.cname == \"signed char\"\n",
        "assert [n for n in dir(p) if not n.startswith('_')] == ['cname', 'kind']\n",
        "''')\n",
        "_run('''casting_to_a_signed_char_truncates_and_compares_by_value''', r'''",
        "p = new_primitive_type(\"signed char\")\n",
        "x = cast(p, -65 + 17*256)\n",
        "assert repr(x) == \"<cdata 'signed char' -65>\", repr(x)\n",
        "assert repr(type(x)) == \"<class '_cffi_backend._CDataBase'>\", repr(type(x))\n",
        "assert int(x) == -65\n",
        "x = cast(p, -66 + (1<<199)*256)\n",
        "assert repr(x) == \"<cdata 'signed char' -66>\", repr(x)\n",
        "assert int(x) == -66\n",
        "assert (x == cast(p, -66)) is True\n",
        "assert (x != cast(p, -66)) is False\n",
        "q = new_primitive_type(\"short\")\n",
        "assert (x == cast(q, -66)) is True\n",
        "assert (x != cast(q, -66)) is False\n",
        "raises(TypeError, sizeof, 42.5)\n",
        "assert sizeof(new_primitive_type(\"short\")) == 2\n",
        "''')\n",
        "_run('''integer_ctypes_wrap_at_their_own_width''', r'''",
        "for name in ['signed char', 'short', 'int', 'long', 'long long']:\n",
        "    p = new_primitive_type(name)\n",
        "    size = sizeof(p)\n",
        "    lo = -(1 << (8*size-1))\n",
        "    hi = (1 << (8*size-1)) - 1\n",
        "    assert int(cast(p, lo)) == lo, name\n",
        "    assert int(cast(p, hi)) == hi, name\n",
        "    assert int(cast(p, lo - 1)) == hi, name\n",
        "    assert int(cast(p, hi + 1)) == lo, name\n",
        "    raises(TypeError, cast, p, None)\n",
        "    assert int(cast(p, b'\\x08')) == 8\n",
        "    assert int(cast(p, '\\x08')) == 8\n",
        "for name in ['char', 'short', 'int', 'long', 'long long']:\n",
        "    p = new_primitive_type('unsigned ' + name)\n",
        "    size = sizeof(p)\n",
        "    hi = (1 << (8*size)) - 1\n",
        "    assert int(cast(p, 0)) == 0\n",
        "    assert int(cast(p, hi)) == hi, name\n",
        "    assert int(cast(p, -1)) == hi, name\n",
        "    assert int(cast(p, hi + 1)) == 0, name\n",
        "    assert int(cast(p, b'\\xFE')) == 254\n",
        "    assert int(cast(p, '\\xFE')) == 254\n",
        "''')\n",
        "_run('''float_and_character_ctypes_round_trip''', r'''",
        "INF = 1E200 * 1E200\n",
        "for name in [\"float\", \"double\"]:\n",
        "    p = new_primitive_type(name)\n",
        "    assert bool(cast(p, 0)) is False\n",
        "    assert bool(cast(p, INF)) is True\n",
        "    assert bool(cast(p, -INF)) is True\n",
        "    assert int(cast(p, -150)) == -150\n",
        "    assert int(cast(p, 61.91)) == 61\n",
        "    assert float(cast(p, 1.25)) == 1.25\n",
        "    assert float(cast(p, b'\\x09')) == 9.0\n",
        "    assert float(cast(p, '\\x09')) == 9.0\n",
        "    assert float(cast(p, True)) == 1.0\n",
        "    raises(TypeError, cast, p, None)\n",
        "p = new_primitive_type(\"char\")\n",
        "assert bool(cast(p, b'A')) is True\n",
        "assert bool(cast(p, b'\\x00')) is False\n",
        "assert cast(p, b'A') == cast(p, 65)\n",
        "assert int(cast(p, b'A')) == 65\n",
        "assert repr(cast(p, b'A')) == \"<cdata 'char' b'A'>\", repr(cast(p, b'A'))\n",
        "''')\n",
        "_run('''pointer_ctypes_are_memoised_and_spell_their_own_name''', r'''",
        "p1 = new_primitive_type(\"int\")\n",
        "p2 = new_pointer_type(p1)\n",
        "assert repr(p2) == \"<ctype 'int *'>\", repr(p2)\n",
        "assert p2 is new_pointer_type(p1)\n",
        "assert repr(new_pointer_type(p2)) == \"<ctype 'int * *'>\", repr(new_pointer_type(p2))\n",
        "assert p2.kind == \"pointer\"\n",
        "assert p2.cname == \"int *\"\n",
        "assert p2.item is p1\n",
        "assert [n for n in dir(p2) if not n.startswith('_')] == ['cname', 'item', 'kind']\n",
        "''')\n",
        "_run('''newp_owns_one_int_and_indexes_only_zero''', r'''",
        "BInt = new_primitive_type(\"int\")\n",
        "raises(TypeError, newp, BInt)\n",
        "BPtr = new_pointer_type(BInt)\n",
        "p = newp(BPtr)\n",
        "assert repr(p) == \"<cdata 'int *' owning 4 bytes>\", repr(p)\n",
        "assert p[0] == 0\n",
        "p = newp(BPtr, None)\n",
        "assert p[0] == 0\n",
        "p = newp(BPtr, 5000)\n",
        "assert p[0] == 5000\n",
        "raises(IndexError, lambda: p[1])\n",
        "raises(IndexError, lambda: p[-1])\n",
        "p[0] = -12\n",
        "assert p[0] == -12\n",
        "q = cast(BPtr, p)\n",
        "assert repr(q).startswith(\"<cdata 'int *' 0x\"), repr(q)\n",
        "assert q[0] == -12\n",
        "''')\n",
        "_run('''the_void_ctype_can_be_neither_cast_to_nor_instantiated''', r'''",
        "p = new_void_type()\n",
        "assert p is new_void_type()\n",
        "assert p.kind == \"void\"\n",
        "assert p.cname == \"void\"\n",
        "assert [n for n in dir(p) if not n.startswith('_')] == ['cname', 'kind']\n",
        "raises(TypeError, newp, p, None)\n",
        "raises(TypeError, cast, p, 42)\n",
        "assert sizeof(new_pointer_type(p)) == sizeof(new_pointer_type(new_primitive_type(\"int\")))\n",
        "''')\n",
        "_run('''array_ctypes_nest_their_names_and_refuse_an_overflowing_length''', r'''",
        "p = new_primitive_type(\"int\")\n",
        "raises(TypeError, new_array_type, new_pointer_type(p), \"foo\")\n",
        "p1 = new_array_type(new_pointer_type(p), None)\n",
        "assert repr(p1) == \"<ctype 'int[]'>\", repr(p1)\n",
        "assert p1 is new_array_type(new_pointer_type(p), None)\n",
        "assert repr(new_pointer_type(p1)) == \"<ctype 'int(*)[]'>\", repr(new_pointer_type(p1))\n",
        "p1 = new_array_type(new_pointer_type(p), 42)\n",
        "p2 = new_array_type(new_pointer_type(p1), 25)\n",
        "assert repr(p2) == \"<ctype 'int[25][42]'>\", repr(p2)\n",
        "assert repr(new_pointer_type(p2)) == \"<ctype 'int(*)[25][42]'>\", repr(new_pointer_type(p2))\n",
        "raises(OverflowError, new_array_type, new_pointer_type(p), sys.maxsize+1)\n",
        "raises(OverflowError, new_array_type, new_pointer_type(p), sys.maxsize // 3)\n",
        "assert p1.kind == \"array\"\n",
        "assert p1.cname == \"int[42]\"\n",
        "assert p1.item is p\n",
        "assert p1.length == 42\n",
        "assert new_array_type(new_pointer_type(p), None).length is None\n",
        "assert [n for n in dir(p1) if not n.startswith('_')] == ['cname', 'item', 'kind', 'length']\n",
        "''')\n",
        "_run('''array_cdata_reads_writes_iterates_and_bounds_check''', r'''",
        "p = new_primitive_type(\"int\")\n",
        "LENGTH = 1423\n",
        "p1 = new_array_type(new_pointer_type(p), LENGTH)\n",
        "a = newp(p1, None)\n",
        "assert repr(a) == \"<cdata 'int[%d]' owning %d bytes>\" % (LENGTH, LENGTH*4), repr(a)\n",
        "assert len(a) == LENGTH\n",
        "for i in range(LENGTH):\n",
        "    assert a[i] == 0\n",
        "raises(IndexError, lambda: a[LENGTH])\n",
        "raises(IndexError, lambda: a[-1])\n",
        "for i in range(LENGTH):\n",
        "    a[i] = i * i + 1\n",
        "for i in range(LENGTH):\n",
        "    assert a[i] == i * i + 1\n",
        "\n",
        "pu = new_array_type(new_pointer_type(p), None)\n",
        "raises(TypeError, newp, pu)\n",
        "raises(ValueError, newp, pu, -1)\n",
        "assert len(newp(pu, 42)) == 42\n",
        "a = newp(pu, list(range(100, 110)))\n",
        "assert len(a) == 10\n",
        "for i in range(10):\n",
        "    assert a[i] == 100 + i\n",
        "raises(IndexError, lambda: a[10])\n",
        "assert list(a) == list(range(100, 110))\n",
        "\n",
        "p2 = new_array_type(new_pointer_type(p), 42)\n",
        "a = newp(p2, list(range(100, 110)))\n",
        "for i in range(10):\n",
        "    assert a[i] == 100 + i\n",
        "for i in range(10, 42):\n",
        "    assert a[i] == 0\n",
        "''')\n",
        "_run('''pointer_arithmetic_and_alignment''', r'''",
        "p = new_primitive_type(\"int\")\n",
        "p1 = new_array_type(new_pointer_type(p), 5)\n",
        "p2 = new_array_type(new_pointer_type(p1), 3)\n",
        "a = newp(p2, [[1, 2], [3, 4], [5, 6]])\n",
        "assert a[0][0] == 1\n",
        "assert a[2][1] == 6\n",
        "assert repr(a + 1).startswith(\"<cdata 'int(*)[5]' 0x\"), repr(a + 1)\n",
        "BInt = new_primitive_type(\"int\")\n",
        "assert alignof(BInt) == sizeof(BInt)\n",
        "BPtr = new_pointer_type(BInt)\n",
        "assert alignof(BPtr) == sizeof(BPtr)\n",
        "assert alignof(new_array_type(BPtr, None)) == alignof(BInt)\n",
        "''')\n",
        "_run('''string_and_unpack_read_a_char_and_int_array''', r'''",
        "BChar = new_primitive_type(\"char\")\n",
        "BArray = new_array_type(new_pointer_type(BChar), 10)\n",
        "a = newp(BArray, b\"hello\")\n",
        "assert len(a) == 10\n",
        "assert string(a) == b\"hello\"\n",
        "BInt = new_primitive_type(\"int\")\n",
        "BIntArray = new_array_type(new_pointer_type(BInt), 5)\n",
        "b = newp(BIntArray, [10, 20, 30, 40, 50])\n",
        "assert unpack(b, 5) == [10, 20, 30, 40, 50]\n",
        "''')\n",
        "_run('''the_module_publishes_its_abi_and_common_type_tables''', r'''",
        "FFI_DEFAULT_ABI\n",
        "FFI_CDECL\n",
        "RTLD_LAZY\n",
        "RTLD_NOW\n",
        "RTLD_GLOBAL\n",
        "RTLD_LOCAL\n",
        "d = {}\n",
        "_get_common_types(d)\n",
        "assert d[\"bool\"] == \"_Bool\", d.get(\"bool\")\n",
        "''')\n",
        "if _failures:\n",
        "    raise AssertionError('; '.join(_failures))\n",
    );
    run_on_worker(PROGRAM, "cffi_backend_m1");
}
