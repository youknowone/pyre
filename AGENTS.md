# AGENTS.md

## Data structure parity with RPython/PyPy

**Do not casually introduce `HashMap` (or any Rust-native collection) when porting RPython/PyPy code.**

majit and pyre are line-by-line ports. The data structure choice is part of
the port — it must match what RPython/PyPy actually uses, even when a Rust
collection looks more convenient.

### Rules

1. **Look up the RPython source first.** Before adding `HashMap`, `HashSet`,
   `BTreeMap`, etc., find the corresponding RPython attribute and check what
   container it uses (`dict`, `list`, an attribute on a class instance, a
   field on `_forwarded`, …). Port that exact shape.

2. **Side-tables are usually wrong.** RPython optimizers store information
   *on the box itself* via `box._forwarded` / `PtrInfo` / `IntBound` /
   descr attributes. If you find yourself reaching for
   `HashMap<OpRef, Something>` to track a per-box property, that is almost
   always a sign you skipped the proper PtrInfo / forwarded slot and are
   inventing a parallel store that RPython does not have. Stop and route
   the data through the existing forwarded/PtrInfo machinery instead.

3. **Borrow-checker workarounds must be minimal and documented.** A
   `HashMap` introduced purely because the borrow checker rejected a more
   direct port is acceptable only when (a) every alternative has been
   tried, (b) the deviation is the smallest possible, and (c) a comment
   cites the RPython original it stands in for. See the
   "majit ↔ RPython Parity Rules" section in `~/.claude/CLAUDE.md`.

4. **Removing an RPython method to "simplify" things is not allowed.**
   If `optimizer.py` defines `ensure_ptr_info_arg0`, the Rust port has
   `ensure_ptr_info_arg0`. Do not delete it because callers can be
   rewritten to a shortcut — the shortcut diverges from RPython and the
   next porter will have no idea why their `heap.py` line-by-line port
   no longer compiles.

### Why

We have already been bitten by this. A previous change deleted
`ensure_ptr_info_arg0` and replaced `arrayinfo.lenbound.make_gt_const(...)`
with a side-table `OptHeap.array_min_lengths: HashMap<OpRef, i64>`. The
side-table then could not be read by `postprocess_arraylen_gc`, so that
function was crippled to a hardcoded `IntBound::nonnegative()`, which then
forced `ExportedValueInfo` to grow a parallel `int_lower_bound` field.
One non-orthodox `HashMap` cascaded into four files of divergence from
RPython. Don't start the cascade.

### When in doubt

Grep RPython:

```
rg -t py 'lenbound|getlenbound|_x86_arglocs|_ll_loop_code' rpython/jit/
```

If RPython stores it on an object attribute, store it on the equivalent
Rust struct field. If RPython stores it on `box._forwarded`, route it
through `OptContext::with_intbound_mut` / `set_ptr_info` / etc. Reach
for `HashMap` only after you have proven that RPython itself uses a
dict-like container in that exact spot.
