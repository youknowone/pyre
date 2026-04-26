# Leaf Porting Work Units

This file tracks the remaining non-parity leaf implementations in the
translation driver stack. Each item names the RPython source location,
the current local leaf, the owning lane, and the acceptance point that
lets another lane build on it.

## Process / Platform / udir

- RPython: `rpython/translator/goal/unixcheckpoint.py:3-77`
- Local leaves: `translator/goal/unixcheckpoint.rs::restart_process`,
  `restartable_point`
- Owner lane: Process/Platform/udir
- Acceptance: `restartable_point(Some("run"))` performs the POSIX fork
  checkpoint path instead of returning the structural `TaskError`.
  Unsupported platforms must follow the upstream nofork branch.

## C Backend Builder

- RPython: `rpython/translator/c/genc.py:40-510`,
  `rpython/translator/c/dlltool.py:7-40`
- Local leaves: `CBuilder::build_database`, `CBuilder::generate_source`,
  `CBuilder::compile`, `CStandaloneBuilder::getentrypointptr`,
  `CStandaloneBuilder::cmdexec`, `CLibraryBuilder::getentrypointptr`,
  `CLibraryBuilder::compile`
- Owner lane: C Backend Builder
- Acceptance: `TranslationDriver::task_database_c`, `task_source_c`, and
  `task_compile_c` can progress through the builder layer without
  builder-level structural `TaskError`s on a simple translated graph.

## Backend Optimizations

- RPython: `rpython/translator/backendopt/all.py:35-164`
- Local leaf: `translator/backendopt/all.rs::backend_optimizations`
- Owner lane: Backendopt
- Acceptance: simple graphs execute the upstream phase skeleton:
  backendopt config copy/set, graph selection, no-op cleanup,
  available constfold/rewrite passes, final `checkgraph`. Missing
  subpasses must be isolated behind their own leaf-level errors.

## LLInterpreter

- RPython: `rpython/rtyper/llinterp.py:28-1477`
- Local leaf: `translator/rtyper/llinterp.rs::LLInterpreter::eval_graph`
- Owner lane: LLInterpreter
- Acceptance: straight-line lltyped graphs with supported operations can
  be evaluated through an `LLFrame`-style loop. Unsupported operations
  should fail at opcode granularity, not at the whole `eval_graph` leaf.

## TargetSpec Callable

- RPython: `rpython/translator/driver.py:573-602`
- Local leaf: `TranslationDriver::from_targetspec`
- Owner lane: TargetSpec
- Acceptance: typed target callable is invoked under timer events, its
  result is unpacked as one of `entry`, `(entry, inputtypes)`, or
  `(entry, inputtypes, policy)`, and `driver.setup(..., extra=...,
  empty_translator=...)` is called.

## Driver C Pipeline Integration

- RPython: `rpython/translator/driver.py:408-541`
- Local leaves: `task_database_c`, `task_source_c`, `task_compile_c`,
  `ProfInstrument::first`
- Owner lane: Driver integration after C Backend Builder
- Acceptance: driver tasks are thin orchestration over the real builder
  methods and preserve upstream state writes: `translator.frozen`,
  `self.cbuilder`, `self.database`, and `self.c_entryp`.

## Instrumentation

- RPython: `rpython/translator/driver.py:44-60`,
  `rpython/translator/driver.py:218-248`
- Local leaves: `ProfInstrument::{first,probe,after}`,
  `TranslationDriver::instrument_result`
- Owner lane: Instrumentation after C Backend Builder and
  Process/Platform/udir
- Acceptance: parent process forks, child enables instrumentation and
  drives compile, parent reads the unsigned-long counter file from
  `udir`.

## Stack Check Insertion

- RPython: `rpython/translator/driver.py:388-392`,
  `rpython/translator/transform.py` `insert_ll_stackcheck`
- Local leaf: `TranslationDriver::task_stackcheckinsertion_lltype`
- Owner lane: Stackcheck
- Acceptance: stack checks are inserted into eligible lltype graphs and
  the driver logs the upstream count.

## JIT Apply

- RPython: `rpython/translator/driver.py:347-363`,
  `rpython/jit/metainterp/warmspot.py`
- Local leaf: `TranslationDriver::task_pyjitpl_lltype`
- Owner lane: JIT Apply
- Acceptance: driver builds the JIT policy, resolves
  `translation.jit_backend`, and invokes an apply-jit hook without
  creating a cyclic crate dependency.

## Jittest

- RPython: `rpython/translator/driver.py:365-376`,
  `rpython/jit/tl/jittest.py`
- Local leaf: `TranslationDriver::task_jittest_lltype`
- Owner lane: Jittest after LLInterpreter, JIT Apply, and unixcheckpoint
- Acceptance: checkpoint runs first, then the jittest module exercises
  the llgraph/JIT path through the driver.
