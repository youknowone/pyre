import _imp


code = compile("x = 1", "orig.py", "exec")
_imp._fix_co_filename(code, "x\udcff.py")
assert code.co_filename == "x\udcff.py"

plain = compile("x = 1", "orig.py", "exec")
_imp._fix_co_filename(plain, "renamed.py")
assert plain.co_filename == "renamed.py"

nested = compile("def f():\n    return 1\n", "orig.py", "exec")
_imp._fix_co_filename(nested, "x\udcff.py")
child = next(const for const in nested.co_consts if isinstance(const, type(nested)))
assert child.co_filename == "x\udcff.py"

realized = compile("def f():\n    return 1\n", "orig.py", "exec")
realized_child = next(
    const for const in realized.co_consts if isinstance(const, type(realized))
)
_imp._fix_co_filename(realized, "x\udcff.py")
assert realized_child.co_filename == "x\udcff.py"

replaced = code.replace(co_filename="y\udcfe.py")
assert replaced.co_filename == "y\udcfe.py"
assert code.replace().co_filename == "x\udcff.py"

replace_parent = compile("def f():\n    return 1\n", "orig.py", "exec")
replace_parent = replace_parent.replace(co_filename="only-parent\udcfc.py")
replace_child = next(
    const
    for const in replace_parent.co_consts
    if isinstance(const, type(replace_parent))
)
assert replace_parent.co_filename == "only-parent\udcfc.py"
assert replace_child.co_filename == "orig.py"

constructed = type(code)(
    code.co_argcount,
    code.co_posonlyargcount,
    code.co_kwonlyargcount,
    code.co_nlocals,
    code.co_stacksize,
    code.co_flags,
    code.co_code,
    code.co_consts,
    code.co_names,
    code.co_varnames,
    "z\udcfd.py",
    code.co_name,
    code.co_qualname,
    code.co_firstlineno,
    code.co_linetable,
    code.co_exceptiontable,
    code.co_freevars,
    code.co_cellvars,
)
assert constructed.co_filename == "z\udcfd.py"

print("OK")
