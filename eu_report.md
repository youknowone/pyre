# The PyPy EU Reports (2004–2007): A Digest

This document summarizes the 31 public deliverables of the EU-funded PyPy project
(IST FP6-004779, "Researching a Highly Flexible and Modular Language Platform and
Implementing it by Leveraging the Open Source Python Language and Community",
1 December 2004 – 31 March 2007, 28 months). It reports **only what the documents
state** — all judgments of success or failure below are the original authors'
own, quoted or attributed. Analysis of which claims remain valid today is kept
out of this file (see `eu_report_assessment.md`); design consequences for pyre
are in `pyre/design.md`.

This is a *reference* digest, written to be consulted when making design
decisions: mechanisms are described at working technical depth, alternatives
that the authors considered and rejected are recorded alongside what shipped,
and every number the reports give is preserved with its context.

Source: the PyPy `extradoc` repository, `eu-report/` directory (D01–D14
deliverables plus the Final Activity Report); local copy at
`~/Downloads/extradoc-branch-extradoc/eu-report/`. Appendix A maps every
section's deliverable IDs to the exact source PDF filenames.

---
## 1. Project frame and timeline (D14.1, D14.3, D14.4, D14.5, Final Activity Report)

### 1.1 Origins and contractual frame

- Concept began as mailing-list discussions in late 2002, "inspired by but
  also frustrated with the results of two successful Python implementations,
  Psycho [sic] and Stackless." First community sprint: Hildesheim, Germany,
  **February 2003** (mailing list grew 0 → 144 subscribers by 2003-02-28).
  During summer 2003 (third sprint) the team concluded full-time funding was
  needed; the proposal was submitted to the European Commission on
  **31 October 2003**; negotiations ran from March 2004; the funded project
  started **1 December 2004**.
- Contract: Sixth Framework Programme, IST Priority 2, STREP, contract
  IST FP6-004779. Originally 24 months in three internal phases (9+9+6),
  extended by amendment to **28 months**, ending 31 March 2007. Five
  contractual amendments in total; Amendment 4 restructured the original
  **58 deliverables into 21 contractual deliverables**; D14.5 states the final
  shape as "14 work packages and in total 30 deliverables."
- Consortium: DFKI (Saarbrücken, coordinator), AB Strakt → later **Open End**
  (Göteborg), Logilab (Paris), Change Maker (Göteborg), merlinux GmbH
  (Hildesheim), tismerysoft GmbH (Berlin), Heinrich-Heine-Universität
  Düsseldorf, Impara GmbH (Magdeburg, joined later); plus four "Physical
  Person" partners (Laura Creighton, Richard Emslie, Eric van Riet Paap,
  Niklaus Haldimann). Governance: a Management Team at consortium level and a
  **Technical Board** of core developers (Krekel, Rigo, Tismer, Pedroni);
  monthly consortium IRC meetings; weekly developer "pypy-sync" meetings.

### 1.2 Phases and milestones

| Phase | Focus (stated) | Milestone |
|---|---|---|
| 1 (to 31 Aug 2005) | "Building a Novel Language Research Tool" | 0.6 + 0.7 releases: a self-contained novel Python implementation |
| 2 | "High Performance" | 0.9 release: "a 20-fold performance improvement by a series of static and core translation optimizations" |
| 3 | "Validation & Flexibility" | PyPy 1.0: the JIT compiler generator plus validation prototypes |

The reports themselves note the phase boundaries blurred: phase 2 "already
validated a number of architectural aspects and also increased flexibility of
the whole system, originally a topic for phase 3"; the extension compiler
(WP03), configuration (WP13) and validation (WP12) were pulled forward "on
commercial/community demand."

### 1.3 Releases and speed trajectory

| Release | Date | Content (as stated) |
|---|---|---|
| 0.6 / 0.6.1 | 20 May 2005 | Core interpreter on top of CPython; compliance 92.08% total, 88.15% of test modules fully passing (vs CPython 2.3.4 suite); ~80% of built-in types (~100 types) ported; "90% of the core modules of the standard library are supported" |
| 0.7 | 28 Aug 2005 | First *translated*, self-contained pypy-c; whole-program type inference ("the annotator") over RPython; first LLVM backend experiments |
| 0.8 | 3 Nov 2005 | Translatable parser/compiler; more translation aspects measured |
| 0.9 | 26 Jun 2006 | 20× faster than 0.7; stackless features, GC framework woven by transformation, extension compiler, logic variables, weakrefs under custom GCs; "core language tests continue to pass at a rate of 95% or better" |
| 0.99 | 17 Feb 2007 | High-level backends (CLI and others) |
| 1.0 | 27 Mar 2007 | JIT compiler generator, transparent proxies, AOP, taint space |

Speed trajectory as stated across the reports: PyPy-on-CPython ~2000× slower
than CPython → first pypy-c ~200× slower (0.7) → 0.8 roughly 10–20× slower →
0.9 "still 2.5–10 times slower than CPython on popular benchmarks" →
1.0 "runs benchmarks at about half of CPython's speed ... not even considering
JIT technologies which will definitely provide a significant edge." The 1.0
statement is accompanied by: "With more refined garbage collection techniques
we believe that we can get very close to CPython speed."

### 1.4 Scale

- Code base: ~30k LOC (+8k test) at start of funding → ~250 KLOC (60 KLOC
  test, ~3000 automated tests) at 0.9 → **~340k LOC (+82k test)** and
  **11,805 automated tests** at the final report.
- Community: "approximately 300–500 people follow the project closely" at end
  of phase 1; in June 2006 "more than 50 developers are following changes to
  the code base, from which only around 20 people are benefiting from EU
  funding"; developer mailing list crossed 300 subscribers.
- 19 week-long sprints (final-report count) across Europe, the US and Japan,
  every 6–8 weeks; participant counts per sprint ranged 7–24.
- Companies engaging with sprints/workshops (listed in D14.3):
  Hewlett-Packard, Philips Medical System, Canonical, CCP Games, Greenpeace
  International, Iona Technologies, EWT LLC, Next Limit Technologies, IBM.
  One offered a sponsored sprint in exchange for performance work; "PyPy
  could not participate in the sponsored sprint in question due to
  differences in objectives and focus at the time."

### 1.5 Final self-assessment

"After 28 months of intense research, development and management activity,
the final PyPy 1.0 milestone delivered all expected results and fulfilled its
objectives." The JIT compiler generator is named **"the major research result
of the project."** The same documents state the frank limitation: "PyPy's
Python interpreter is still not generally usable. The main blocking factor is
the lack of extension modules," the consequence of a deliberate early
decision — "The project consciously decided early to first aim at realizing
the full vision of PyPy's original claim, targeting the research oriented
results before putting engineering and refactoring efforts into better
extension module support. Indeed PyPy 1.0 has some of the rough edges that
can be expected from the offspring of a research effort like PyPy."

### 1.6 The l × o × p thesis (Final Activity Report)

PyPy's stated purpose: language implementers otherwise write l × o × p
interpreters for l languages, o low-level design decisions (memory
management, threading model, target environment), and p platforms. PyPy makes
each parameter independent: evolve or replace the interpreter (l), tweak the
translation process (o), write new backends (p). Explicit contrast with
standardized VMs: ".NET … enforces p = 1. We believe that enforcing the use
of one common environment is not necessary." The final report's summary
claim: "PyPy breaks the compromise between flexibility, simplicity and speed
for implementing today's dynamic computer languages."

The three publishable/exploitable results the final report names:
1. **The PyPy language implementation platform** — "supporting multiple
   different dynamic languages, with Python, JavaScript and Prolog currently
   implemented"; backends C/POSIX, LLVM (low-level), CLI/.NET, Smalltalk, JVM
   (high-level); "the Python/C/POSIX combination approaching readiness for
   production use"; MIT license.
2. **The flexible Python interpreter** — "a fully compliant interpreter for
   the Python 2.4 language … more flexible and open to language research and
   enhancements than any pre-existing implementation of Python."
3. **The py.test testing framework** — released separately, MIT license, with
   commercial support from merlinux.

---
## 2. The Standard Interpreter (D04.1, D04.2, D04.3, D04.4)

The four WP04 deliverables document releases 0.6–0.8 and constitute the
canonical architecture reference for the interpreter. D04.2 states its own
origin: architecture/coding documentation was bundled into it "due to an
omission in the Description of Work."

### 2.1 Bytecode interpreter / object space split

The Standard Interpreter is divided into two independent subsystems:

- the **bytecode interpreter**: control flow, frames, exceptions, tracebacks,
  and the value stack of *black-box wrapped objects*. It uses "the same
  compact bytecode format as CPython 2.4". Each bytecode is implemented by a
  Python function which delegates operations on application-level objects to
  an object space — "This interpretation and delegation is the core of the
  bytecode interpreter."
- the **object space**: "creates all objects and knows how to perform
  operations on the objects … a library offering a fixed API, a set of
  operations with implementations that correspond to the known semantics of
  Python objects." Example given: `add` performs numeric addition on numbers
  and concatenation on sequences. **All operations take and return wrapped
  objects.** Only a few very simple operations let the interpreter learn
  anything about a value — "The most important one is `is_true()`, which
  returns a boolean interpreter-level value," needed for conditional-branch
  bytecodes.

Four object spaces plug into the same interpreter:

| Space | Role | Notes from the reports |
|---|---|---|
| **Standard** | the real implementation | W_IntObject, W_ListObject, … — "the equivalent of the C structures PyIntObject, PyListObject" |
| **Trace** | records/prints every operation, frame event, bytecode | "The ease of implementation … underlines the power of the Object Space abstraction. Effectively it is a simple proxy object space"; enabled at runtime by `__pytrace__ = 1`; used pedagogically to show how abstract interpretation records execution |
| **Thunk** | lazy values + `become` (global identity exchange) | 100–152 LOC; wraps another space, forcing arguments to operations; "all objects grow a field possibly pointing to an object that should be used instead of them"; translatable since 0.8 |
| **Flow** | records operations into control-flow graphs | the front end of the translator (§3.2); "it is actually just an alternate representation for the function" |

The 0.8 report (D04.4) names the two extensibility axes this architecture
deliberately provides: **syntactic** (the translatable parser/compiler; "hook
at the syntactic level") and **semantic** (object-space delegation; "this
exploits the object space interface as a surface to hook into language
semantics") — both stated as requirements from WP09/WP10 (aspects,
design-by-contract, constraints).

### 2.2 Application level vs interpreter level

A two-level code model with strict conventions (D04.2 §6.6, coding guide):

- `w_xxx` = wrapped (app-level) object; `xxx_w` = interp-level container of
  wrapped objects (a list/dict *of* wrapped objects — not a wrapped
  list/dict); `space` = the object space instance, always passed under that
  name. Wrapped values include `w_self`.
- App-level `a + b` is interp-level `space.add(w_a, w_b)` — the stated
  analogy is CPython's C level (`PyNumber_Add(p_a, p_b)`).
- Never use `w_x == w_y` or `w_x is w_y` ("DON'T DO THAT" — no reason two
  wrappers are related even if they hold the same value); never `if w_x:`
  (an error). Use `space.eq_w`, `space.is_w`, `space.is_true`.
- `space.unwrap` "must be avoided whenever possible … only when you are well
  aware that you are cheating, in unit tests or bootstrapping code"; the
  typed variants (`int_w`, `str_w`, `float_w`, `interpclass_w`,
  `unpackiterable`) are the sanctioned probes.
- All app-level exceptions are carried by **`OperationError`** at interp
  level, sharply distinguishing app-level failures from interpreter-level
  bugs. Raise: `raise OperationError(space.w_XxxError, space.wrap("msg"))`.
  Catch by `e.match(space, space.w_XxxError)` — **not** by comparing
  `e.w_type`, which misses subclasses. A stated future direction (never
  qualified further): replacing OperationError with a family of common
  exception classes (AppKeyError, AppIndexError, … with a generic AppError).
- **"Application level is often preferable"**: app-level code is
  substantially higher-level, easier to write and debug (the worked example
  is `dict.update` — three obvious lines at app level vs a verbose
  `space.call_method`/`space.iter`/`space.next` loop catching StopIteration
  at interp level). "In almost all parts of PyPy, you find application level
  code in the middle of interpreter-level code." The only caveat is
  bootstrapping order (app-level helpers need an initialized space).

**Gateways** cross the barrier in both directions
(pypy/interpreter/gateway.py), described as "somewhat involved, mostly due to
the fact that the type-infering annotator needs to keep track of the types of
objects flowing across those barriers":

- `interp2app(func)` exposes an interp-level function at app level; an
  **`unwrap_spec`** (e.g. `[ObjSpace, W_Root, Arguments]`) declares how
  app-level arguments are unwrapped.
- `app2interp` / `gateway.applevel(...)` let interp-level code call app-level
  helpers transparently (worked example: the metaclass-finding algorithm is
  written at app level and invoked from the `BUILD_CLASS` opcode). The
  stated benefit: the app-level implementation can later be rewritten at
  interp level without changing callers.
- Gateway stubs, frame classes and argument-parsing code are **generated at
  initialization time** "in order to let the annotator only see rather
  static program flows with homogeneous name-value assignments on function
  invocations" — full Python dynamism is allowed at bootstrap and invisible
  to analysis.

Builtin modules are **mixed modules** (`pypy/module/*/__init__.py` declaring
`appleveldefs` and `interpleveldefs` dictionaries; `app_`-prefixed submodules
are app-level; inline interp expressions allowed, e.g.
`'None': '(space.w_None)'`). "There is no extra facility for
pure-interpreter-level modules because we haven't needed it so far."
Pure-Python rewrites go to `pypy/lib/`. Module lookup order:
`pypy/module` (mixed builtins) → PYTHONPATH → `pypy/lib/` →
`lib-python/modified-2.4.1/` → `lib-python/2.4.1/` ("Never ever checkin
anything here" — stdlib modifications are made on an `svn cp`'d copy so the
two directories can be diffed). Modifications were often needed because PyPy
made all classes new-style by default while CPython's stdlib relied on some
being old-style (old-style support existed behind `--oldstyle`, implemented
mostly as user-level code).

### 2.3 Bytecode interpreter internals (D04.2 §7)

- Interpreting a code object = instantiate a **Frame** and call
  `frame.eval()`. Both CPython and PyPy are stack-based VMs. Frame state:
  a "fast scope" array of wrapped locals, a **blockstack** (nested
  control-flow: while/try), the value stack, a reference to the globals
  dict, and traceback debugging info.
- PyPy constructs **four specialized frame classes**, chosen per code
  object: `PyInterpFrame` (plain), `PyNestedScopeFrame`, `PyGeneratorFrame`,
  `PyNestedScopeGeneratorFrame` (inheriting from both) — specialization
  generated at initialization time, invisible to the annotator.
- **Code** objects mirror CPython attribute-for-attribute (co_flags,
  co_stacksize, co_code, co_argcount, co_varnames, co_nlocals, co_names,
  co_consts, co_cellvars, co_freevars, co_filename, co_firstlineno, co_name,
  co_lnotab) and create frames via `create_frame()` — "with proper support of
  parser and compiler this should allow to create custom Frame objects
  extending the execution of functions in various ways" (already used for
  generators and nested scopes).
- **Function** carries func_doc/name/code/defaults/dict/globals/closure and
  a `__get__` descriptor producing **Method**; both execute through
  `call_args()` taking an **Arguments** instance (argument.py), which owns
  positional/keyword/default/star/star-keyword binding and error reporting —
  "Function argument parsing is a significant complexity," the stated reason
  for the generated specialized code.
- **Module** and **MixedModule** classes; `__builtin__` is the example mixed
  module.
- Introspection is descriptor-based: Function, Code, Frame, Module are all
  **`Wrappable`**; a space asks a wrapped object for its type via
  `getclass`, then calls the type's `lookup(name)` for the descriptor
  (typedef.py). The reports cite Raymond Hettinger's descriptor how-to as
  the model.

The **object space interface** (D04.2 §8.2, "a draft version … still
evolving, although the public interface is not evolving as much as the
internal interface") consists of: administrative methods (`initialize`,
`getexecutioncontext`); the full operation table mirroring CPython's abstract
object interface (id, type, issubtype, iter, repr, str, len, hash,
getattr/setattr/delattr, getitem/setitem/delitem, unary and binary arithmetic
including all `inplace_*` variants, comparisons, `contains`, descriptor
get/set/delete, `next` — which raises a real NoValue at exhaustion — `call`,
`call_function`, `is_`, `isinstance`, `exception_match`); object creation
(`wrap`, `newbool`, `newtuple`, `newlist`, `newdict` — which takes an
interp-level list of pairs, not a dict — `newslice`, `newstring`,
`newunicode`); conversions (`unwrap`, `int_w`, `str_w`, `float_w`,
`interpclass_w`, `is_true`, `unpackiterable`); and data members
(space.builtin, space.sys, w_None, w_True, w_False, w_Ellipsis,
w_NotImplemented, plus ObjSpace.MethodTable/BuiltinModuleTable/ConstantTable/
ExceptionTable).

### 2.4 The Standard Object Space

- "The direct equivalent of CPython's object library (the 'Objects/'
  subdirectory)": abstract parent `W_Object`, subclasses `W_IntObject`,
  `W_ListObject`, … A wrapped object is a black box for the main loop.
- **Everything is wrapped, including integers.** The rationale is stated at
  length: using plain host integers would force case-testing everywhere
  (the analogy drawn is CPython's `PyObject*`-except-ints or odd-pointer
  small-int tricks, which "puts extra burden on the whole C code");
  "wrapping integers as instances is the simple path, while using plain
  integers instead is the complex path, not the other way around"; plain-int
  representation "is a later optimization … it could be introduced by the
  code generators at translation time" (which is what the tagged-pointer
  work in §4 later did).
- **Multimethod dispatch** replaces CPython's `Object/abstract.c` rules.
  Implementations are named per argument classes (`add__Int_Int(space, w1,
  w2)` — the worked example uses `.intval`, `ovfcheck(x+y)`, and raises
  `FailedToImplement(space.w_OverflowError, …)` to fall through to the next
  candidate). *Delegate* functions convert between implementations
  (e.g. int→float). Python-level `__add__`/`__radd__` methods are produced
  by **slicing** the multimethod tables on the first/second argument,
  reproducing CPython object-model corner cases (`NotImplemented` returns).
  The stated translation hope: multimethods "can probably be translated to a
  different low-level dispatch implementation that would be binary
  compatible with CPython's (basically the PyTypeObject structure and its
  function pointers)," or "more straightforwardly converted into some
  efficient multimethod code" if compatibility is not required.
- **Two modules per type**: the type-specification module `xxxtype.py`
  (the user-visible type object; e.g. listtype.py enumerates list methods)
  and the implementation module `xxxobject.py` (data storage + operations);
  `__new__()` locates the implementation and instantiates it. The declared
  goal: "It is possible (though not done so far) to provide several
  implementations of the instances of the same Python type. The `__new__()`
  method could decide to create one or the other" — invisible to the user.
  A `typedef` class attribute points an implementation back to its type
  specification. (D06.1's multidicts/multilists/string-variants, §4, are
  this mechanism exercised.)

### 2.5 RPython as seen from the coding guide (D04.2 §10)

"We have no formal language definition as we think it is more practical to
discuss and evolve the set of restrictions while working on the whole
program analysis." Unlike source-to-source translators (Starkiller is the
named contrast), analysis starts from **live code objects** after bootstrap
(§3.1). The rules the coding guide fixes:

- **Flow restrictions**: a variable holds values of at most one type at each
  control-flow point (None mixed with instances is allowed as null pointer);
  module globals are constants; all control structures allowed except
  `yield` (no generators); no run-time definition of classes or functions;
  exceptions fully supported under the rules below.
- **Object restrictions**: no variable-length tuples (each element-type ×
  length combination is a distinct type); lists are allocated arrays
  (over-allocated so append is fast) with only common index/slice forms
  allowed (bound-checked only when an IndexError handler is present; no
  step; negative indexes restricted); dicts need a unique hashable key type
  (historically string-only, later generalized — with the note that "the
  implementation could safely decide that all string dict keys should be
  interned"); methods and class attributes don't change after startup;
  statically-called functions may use defaults and varargs but dynamic
  dispatch "enforces very simple, uniform signatures (currently only in
  opcode dispatch)"; `len` is a special form (a structure never measured by
  len may drop its length field).
- **Integer model**: machine-sized signed arithmetic with silent wrap-around
  after translation, plus explicit helpers: `ovfcheck()` (wraps a single
  arithmetic op; pre-translation it detects int→long promotion, post-
  translation it becomes one overflow-checking C operation),
  `ovfcheck_lshift()`, `intmask()` (wrap-around; code generators ignore it
  entirely), and `r_uint` (a pure-Python word-sized unsigned class carried
  through annotation; mixing signed with r_uint yields unsigned). "We have
  no equivalent of the 'int' versus 'long int' distinction of C at the
  moment and assume 'long ints' everywhere."
- **Exception rules**: by default, simple operations raise no exceptions
  after translation *unless an exception handler is present* — "supplying an
  exception handler is how you ask for error checking," and its absence is
  an assertion the operation cannot fail. **The rule does not apply to
  calls: any called function is assumed to be allowed to raise any
  exception.** Explicit raises are always generated. Running on CPython
  keeps all checks live — "PyPy is Debuggable on Top of CPython," catching
  violated implicit assertions.

### 2.6 Parser and compiler (D04.3, D04.4)

Requirements stated three times: **flexible, translatable, extensible** —
"extend the grammar for more language features but still using existing
bytecode" and "eventually extend the bytecode generation," driven by WP09/
WP10 needs.

The history is a sequence of explicit pivots (each with its stated reason):

1. **First version**: regex tokenizer; grammar parser produced a tree of
   nested tuples fed to CPython's pure-Python `compiler` package.
2. **Second version**: (a) tokenizer replaced by **Jonathan David Riehl's
   DFA/automata tokenizer** — "much easier to convert to RPython than the
   original tokenizer (at that time the regular expression module was not
   written in RPython yet)"; (b) "nested tuples of variable length cannot be
   translated with PyPy," so a **StackElement** composite (terminal/
   non-terminal instances) mirrored the tuple tree "in an RPythonic way,"
   with the still-untranslatable compiler converting back to tuples.
3. **Third version — AstBuilder**: builds the AST *directly during parsing*,
   dropping CPython's Transformer, which was "buggy, complex and … difficult
   to translate to RPython." The precise typing obstruction is recorded: the
   transformer input tuples `('rulename', arg-or-token, lineno)` cannot be
   RPython-typed because the second slot is heterogeneous, and "the path we
   chose was to produce AST directly with the AST builder." Noted structural
   difficulty of the direct approach: AstBuilder reduces bottom-up, so
   context is sometimes missing (`a.b` is a different node on the left vs
   right of an assignment); resolved by post-modification or deferred node
   instantiation.

Architecture: three independent actors — **Tokenizer** (DFA-based;
Memento-pattern `context()`/`restore()` for backtracking, since "Python's
grammar is mostly LL(1)" but not strictly), **Parser** (grammar rules as a
**Composite** of GrammarElement subclasses: Sequence, Alternative,
KleeneStar, Token, each with recursive-descent `match()`), and **Builder**
(interchangeable; the source of the version history above). "The three parts
we just described are really independent, and that's one of the reasons why
our parser is so flexible."

The Python grammar is **built at startup from CPython's pristine `Grammar`
file** through a grammar-of-grammars (the Grammar file's own syntax "never
changes across versions," so its parser is defined once; a two-pass
EBNFVisitor links rule symbols and creates anonymous `:`-prefixed subrules).
Motivation: "As the syntax is likely to change between different versions of
Python, it is important for PyPy to have an automated way to build a correct
grammar object depending on the version of Python we want to use."

The compiler is a **~50% rewrite of CPython's `compiler` package** (which
CPython itself never uses for normal compilation). ASTs are traversed twice
— once to gather scoping information, once to emit a flow graph of bytecode.
RPython-driven changes: the dynamic visitor (`getattr(visitor, "visit_%s" %
classname)`) was replaced by a generated static `accept()` per node class;
state passed down the tree via variable signatures was rewritten as stacks on
the visitor. "In the process of porting this original package to RPython, we
discovered a number of unexpected problems and bugs. This made the porting
more involved and time-consuming than expected"; comparisons against either
CPython compiler were judged unreliable (the Python package "has a lot of
bugs in corner cases"; the C compiler applies optimizations "for which we
don't care for now"), so the operational criterion chosen was: "we chose to
consider our compiler operational as long as all the compliance tests pass."

Bytecode assembly: **FlowGraph/PyFlowGraph** of branchless Blocks;
`flattenGraph` topologically sorts to minimize jumps, then patches jump
targets. Recorded limitation: forward long jumps (>65435 bytes) are not
implemented — "long jumps … can only be made backward" — justified by
CPython parity, the rarity of such code objects, and "a sure sign of bad
programming practice." The line-number table uses CPython's compressed
two-byte encoding. All instructions were wrapped in `Instr` instances for
RPython compliance.

Stated future evolutions (D04.3 §7): static syntax experiments by editing
the grammar file; **dynamic** syntax modification ("an Oz-like syntax for
logic/constraint programming triggered by `import csp_syntax`"); AST
rewriting before compilation (optimizations, AOP/DbC injection); extending
bytecodes judged "certainly not trivial" and probably unneeded.

### 2.7 Testing conventions established by WP04

Two kinds of unit tests: **interp-level tests** (run on CPython against a
space) and **app-level tests** (`AppTest*`, run by PyPy's interpreter;
cannot import interp-level modules; data passed via `setup_class` and
`w_`-prefixed class attributes). The `-o` switch selects the object space.
Compliance tests are CPython's regression suite with modified copies kept
under `lib-python/modified-X.Y.Z/` so the delta against the pristine tree is
diffable. An issue tracker (roundup) is driven from svn commit messages.

---
## 3. RPython and the translation toolchain (D05.1–D05.4, D07.1)

D05.1 ("Compiling Dynamic Language Implementations", 43 pp) is the
theoretical core of the corpus — flow space, annotator with a formal model
and proofs, RTyper, backends. D05.2 is the 0.7 release report; D05.3 covers
memory management and threading as translation aspects; D05.4 is the
overview paper with the measured aspect costs; D07.1 (§4 below) finishes the
GC and stackless stories.

### 3.1 The core idea: analysing live programs

- Full static analysis of Python is framed as impossible in principle: "The
  notion of 'declaration', central in compiled languages, is entirely missing
  in Python." Class/function/module definitions are *statements executed at
  runtime*; `os.py` builds its interface dynamically per host OS; import
  hooks are common. "This point of view should help explain why analysis of
  a program is theoretically impossible: there is no declared structure to
  analyse" (D05.1 §3.1). The authors consider this more fundamental than the
  classical `eval`/introspection arguments.
- The pivot: analyse **live programs in memory** after bootstrap. "In some
  sense, we use the full Python as a preprocessor for a subset of the
  language, called RPython." Three levels of restriction are contrasted:
  analysing dead source (gives up all dynamism); analysing a frozen memory
  image (gives up dynamism after a point — "natural in Smalltalk-like image
  environments"); PyPy goes further and allows fully dynamic sections *as
  long as they are entered a bounded number of times*. Worked example:
  `interpreter/gateway.py` builds a custom wrapper class per function — a
  finite set, but one that is hard to enumerate manually, so **the inference
  tool itself invokes the class-building code as part of inference**.
- RPython is deliberately **not formally defined**: "we have no formal
  language definition as we think it is more practical to discuss and evolve
  the set of restrictions while working on the whole program analysis"; its
  definition is informally "Python without the operations and effects that
  are not supported by our analysis toolchain." Being RPython is a property
  of *entire programs* (the annotator is a global analysis). "It is mainly
  because of this trade-off situation [flow space vs annotator capability]
  that the definition of RPython has shifted over time" (D04.2).
- Lineage from Psyco's dynamic analysis: the ability to "fall back to
  regular interpretation for parts that cannot be understood is a central
  feature."

### 3.2 The Flow Object Space (D05.1 §5)

The *unmodified* bytecode interpreter runs with a special object space whose
objects are empty placeholders; recorded operations give "an assembler-like
view of what the function performs." "An object space is thus an
interpretation domain; the Flow Object Space is an abstract interpretation
domain." The flow space is described as a "short but generic plug-in" — it
"enables an interpreter for any language to work as a front-end", and syntax
or bytecode changes need implementing only once, in the standard
interpreter.

Mechanics as documented:

- The flow space interrupts the interpreter after every bytecode and
  synthesizes a **frame state** (position-dependent data: bytecode index,
  exception-handler stack; plus the flattened list of variables/constants),
  comparing it with previously seen states at the same position. New state →
  new block; identical state → backlink (loop closed); "similar enough"
  (same position-dependent part) → **merge** into a more general state via a
  union operation: two equal constants unify to that constant, everything
  else unifies to a fresh variable. If the merged state is strictly more
  general, the existing block is cleared and re-used ("Reusing the block
  avoids the proliferation of over-specific blocks" — e.g. it prevents
  unrolling the first loop iteration with the counter constant).
- **Branching**: `is_true` on a variable must capture both paths. "Without
  proper continuations in Python," an explicit replay scheme is used: normal
  blocks (**SpamBlocks**) carry a frame state; branch outcomes create
  special blocks (**EggBlocks**) with *no* frame state, forming a binary
  tree under the nearest SpamBlock; an EggBlock is reached by **replaying**
  the recording from the root block, with replaying recorders checking that
  the same operations are re-issued and answering each `is_true` per the
  branch. This captures all flow paths *including those internal to the
  interpreter engine*, not just those visible in bytecode — e.g.
  `UNPACK_SEQUENCE n` generates a tree with n+1 branches. Stated
  limitation: the engine may not use an unbounded loop to implement a single
  bytecode; all loops must exist in the bytecode, because backlinks are only
  inserted from the end of one bytecode to the beginning of another.
- **Dynamic merging**: a block is only created at a bytecode that actually
  produces an operation, so merging happens only there — which
  constant-folds aggressively across branches (two syntactically different
  functions can produce the same graph). **Admitted consequence**: the flow
  space "is not guaranteed to terminate" — a constant infinite loop is
  followed forever. The stated mitigation is cultural, not technical: "we
  make sure that the code that we send to the Flow Space is first
  well-tested. This philosophy will be seen again."
- **Geninterp**: app-level helper code is automatically turned into
  interp-level code by flowing it and emitting one `space.xxx()` call per
  recorded operation; the goto-less output uses a `next_block` dispatch
  loop, which dynamic merging later constant-folds away when such code is
  re-analysed.
- Graphs are in **SSI form** (Static Single Information, an extension of
  SSA): every variable is used in exactly one block; live values are passed
  along links like call parameters instead of phi nodes; each block declares
  input variables. Operations are written `z = opname(x1,…,xn)|z'` where z'
  is an auxiliary variable used by special rules (see the list rule below).

### 3.3 The Annotator (D05.1 §6)

Forward abstract interpretation over a lattice, with a formal model and
proofs — the term "annotation" is chosen over "type" because "an annotation
is a set of possible values, and such a set is not always the set of all
objects of a specific Python type."

- **Lattice**: `Bot`/`Top`; `Bool ≤ NonNegInt ≤ Int`; `Char ≤ Str`;
  `Inst(class)` with `Inst(sub) ≤ Inst(base)`; `List(v)` where v is a hidden
  variable summarising the items (List terms mutually unordered);
  `Pbc(set)` (subsets of the finite, discovered-as-you-go set of pre-built
  constants: functions, classes, potential bound methods C.f, frozen
  pre-built constants, None = Pbc({None})); **nullable twins** of
  string/instance/Pbc annotations track what can be NULL after translation
  (all lists implicitly nullable). Every annotation also has a
  single-known-object constant variant. The full extended lattice (Dict,
  Tuple, Float, UnicodePoint, Iterator, …) lives in
  `pypy/annotation/model.py`.
- **Why forward-only**: Python "gives useful info about a variable only from
  how it was produced, not how it is used" — deliberately "a more naive
  approach than … Hindley-Milner." A state is (bindings b: V→A, equivalence
  relation E on V); rules generalise the state monotonically; a fixed point
  is reached because the lattice has no infinite ascending chain.
- **Mutable objects are named the hard part**: "Tracking mutable objects is
  the difficult part of our approach." Lists are homogenised (RPython has no
  heterogeneous lists); aliasing is handled by embedding a **hidden item
  variable** in each `List(v)` annotation, shared by all aliases, so
  generalising an item type instantly reschedules every reader. Merging two
  lists identifies their hidden variables — "This process gradually builds a
  partition of all lists in the program, where two lists are in the same
  part if they are combined in any way."
- **Classes/instances**: attributes are attached to the class where first
  seen written; polymorphic use lifts them to the common base. Stated
  assumptions/limits: "the annotator is limited to single inheritance plus
  simple mix-ins"; all instances reaching a program point must share a
  user-defined common base (not `object`); bound methods may be passed
  around but **not stored back into instances** ("It is a limitation of our
  annotator to not distinguish these two levels — there is only one set of
  v_{C.attr} variables for both" class- and instance-level attributes). A
  `lookup_filter` mechanism recovers precision lost by identifying attribute
  variables up the hierarchy.
- **Calls**: one combined rule per `simple_call` handles Pbc sets mixing
  functions, classes and methods — "Calling a class returns an instance and
  flows the annotations into the constructor __init__."
- **Proofs and cost**: generalisation, termination, and soundness are proved
  for the static model (§6.9; the authors call the goal "an intuitive
  understanding of why annotation works"); with the non-static extensions,
  termination is provable only under "a computable bound on the number of
  functions and classes that can ever exist at run-time" (true for PyPy).
  "No formal complexity bound" is given — "Worst-case scenarios would expose
  severe theoretical problems. In practice, these scenarios are unlikely."
  Empirically: annotating all of PyPy (~20,000 blocks / ~4,000 functions)
  took **~5 minutes**, with each rule re-applied 20–40× — "suggesting
  n·log(n) practical complexity." A stated future need that never shipped in
  the EU period: **modular annotation** (imposing annotations at interface
  boundaries so parts can be annotated independently).
- **Non-static extensions** (§6.10) — "in practice annotation is much less
  'static'":
  - **Specialization** by explicit flags: per-argument-annotation copies;
    per-argument-value; **ignoring** (the call is dropped — used for
    test/debug code); **by arity** (default for `*args`); **ctr_location**
    (a fresh class copy per instantiation site — "a simple but potentially
    over-specializing way to get class polymorphism"); **memo** (the call is
    fully computed at annotation time and compiled as a table lookup).
  - **Concrete-mode execution**: memo functions and everything they call are
    concretely executed during annotation with *no* staticness restriction,
    typically instantiating classes, sometimes building new classes and
    functions — "used quite extensively in PyPy"; "switching to concrete
    mode execution is an integral part of our annotation process."
  - **Constant propagation** exists mainly for **dead-code removal**, not
    speed ("low-level compilers already do this well") — a
    `Bool(const=False)` branch is simply never followed. This is the
    mechanism by which bootstrap-only code is hidden from analysis (frozen
    pre-built constants force their caches, so the `if not_computed_yet:`
    branch is dead).
  - **Narrowing**: `isinstance(obj, Sub)` and comparisons narrow annotations
    per-branch via an extended Bool annotation carrying (true-case,
    false-case) refinements per variable.
- **Error philosophy**: "we do not generally try to prove the correctness
  and safety of the user program, preferring to rely on test coverage" —
  e.g. concatenating two nullable strings is taken as a *hint* they are not
  None, not an error. Degeneration to `Top` is reported at its first
  appearance (only possible once the toolchain matured; Top had been an
  essential fallback during development).

### 3.4 RTyper and the low-level object model (D05.1 §7.1, D05.3 §4)

"The central bridge" between annotator and backends: it emits no source, but
replaces each RPython-level operation with low-level operations, chosen by
**representation objects** keyed on annotations — one representation per
used annotation.

- **Two type systems**: **lltype** — C-like: primitives, structs, arrays,
  function pointers, opaque types, non-primitives handled only via pointers;
  memory management still partially implicit (the backend inserts
  refcounting/GC). **ootype** — classes/instances for OO backends (see
  §6.2). Subclassing under lltype = nested substructures: an instance of B
  is a struct whose first field is the substructure for parent A; "These
  structure layouts are quite similar to how classes are usually implemented
  in C++."
- Documented vtable layout (D05.3 §4): one vtable per class;
  `object_vtable` = {parenttypeptr, rtti, subclassrange_min,
  subclassrange_max, name, instantiate()}; subclass vtables inline the
  parent vtable and append method function pointers and data class
  attributes; instances = {typeptr} plus inlined parent struct plus own
  fields.
- **Subclass checking** is cited as the flexibility show-piece: a naive
  linear parent walk was replaced by the **relative numbering algorithm**
  (subclassrange_min/max) "by changing just the appropriate code of the
  rtyping process."
- **Identity hashes of pre-built constants**: the default identity hash is
  the address, but a PBC's pre-translation hash may already have been
  captured; the solution is an extra per-instance field caching the
  pre-translation address (zero for fresh objects, filled on first hash) —
  "A similar strategy is required, anyway, if we want to use a copying
  garbage collector later on."
- **Cached functions with PBC arguments**: a function from a finite Pbc set
  is executed concretely at annotation time and compiled as a field read on
  the PBCs.
- **Representation choice driven by annotation**: resizable lists get an
  extra indirection (struct → array) with over-allocation; never-resized
  lists become plain arrays; lists known to be unmutated `range()` results
  store only start/stop. The stated plan for **tagged pointers** ("instead
  of boxing … requiring explicit hints on the classes; field access would
  become masking operations") is the origin of the D07.1 experiment (§4).
- **Helpers and LLPython**: any operation with no direct C equivalent gets a
  helper *written in Python* and fed back through flow space → annotator →
  RTyper, with different default specializations at that level (several
  copies per low-level argument type). "This approach shows that our
  annotator is versatile enough to accommodate different kinds of
  sub-languages at different levels … the resulting language feels like a
  basic C++ without any type or template declarations." A D14.2 slide
  records the rejected alternative: "Originally we tried to do the job the
  RTyper does at the same time as source generation. Failed. Miserably."

### 3.5 Backends (D05.1 §7.2, D05.2)

- "Basically straightforward, [but] messy in practice." The backend owns
  memory management, the exception model, and execution-model changes
  (coroutines). Little code is shared even between GenC and GenLLVM, which
  consume the same low-level graphs.
- **GenC** works in two passes: (1) recursively collect all functions and
  pre-built data structures, recording struct types; (2) emit forward struct
  declarations → struct definitions → forward function/data declarations →
  bodies and static initializers. Each block becomes labeled C with jumps;
  a few "primitive" functions have no graph and are hand-written in a
  support C file.
- Another rejected code-generation alternative is recorded: "writing a lot
  of template C code that gets filled with concrete types" was tried and
  abandoned in favour of the RTyper + backend split.
- D05.2 outcome summary for 0.7: toolchain for C and LLVM; "ref counting and
  Boehm garbage collectors," thread models, and platforms (Mac OS X, Linux,
  Windows) selectable modularly; compiled pypy-c "200 times slower than
  CPython and a factor of 10 times faster than pypy running on top of
  CPython. Execution speed was considerably improved shortly after the
  release." Early Java backend work "mothballed"; Impara joined for the
  Squeak backend; JavaScript experiments noted.
### 3.6 Translation aspects — the doctrine and its measured cost (D05.3, D05.4, D07.1)

The reports' central architectural claim: low-level concerns are **not present
in the interpreter source** and are woven in during translation.

- "The implementation code simply makes no reference to memory management …
  This contrasts with CPython where the decision to use reference counting is
  reflected tens or even hundreds of times in each C source file." And, in
  D07.1's restatement: refcounting chosen early "accounts for the fact that
  operations to increase and decrease reference counts are sprinkled over all
  the source files, which makes changing this choice very hard."
- "Any combination of aspects can be selected freely, avoiding the problem of
  combinatorial explosion of variants which can be seen in manually written
  interpreters." (See §6.4 for the 1.0 compatibility matrix that qualifies
  this in practice.)
- **The helper-graph pattern**, used by every transformation: new
  functionality (GC logic, stackless bookkeeping) is written as RPython
  "system code" helpers, sent through the same toolchain to produce extra
  graphs, and "the transformation proper only inserts calls to these extra
  graphs into the original graphs as appropriate."
- The correctness argument is made explicitly for refcounting: CPython
  programmers must make "a large number of not-quite-trivial decisions about
  the refcounting code," and "Experience suggests that mistakes will always
  creep in, leading to crashes or leaks … it is surely better to not write
  the reference count manipulations at all. … Writing the code that emits the
  correct reference count manipulations is surely harder than writing any
  single piece of explicit refcounting code, but once it is done and tested,
  it just works without further effort."
- **Multiple interpreters** come for free: with a single space instance the
  `space` pointer is a pre-built constant and *disappears* from generated
  code (arguments, locals, fields); with two instances a space attribute is
  kept automatically — "the best of both worlds." (CPython's interpreter-state
  API is cited as inherently partial — interned strings are a file-level
  static shared between interpreters, and an `interp` pointer on every object
  was judged too costly there.)
- **Measured aspect costs at 0.8** (D05.4, October 2005; binary ≈5.6 MB on a
  Linux Pentium; "We have not particularly optimized any of these aspects
  yet"): stackless +8% time / +28% size (~300 lines of changes, done at the
  Paris sprint "in a couple of days" — versus "about six months" for the
  original invasive CPython stackless patches); two object-space copies +10%
  / +40%; thunk space +6% / +13%; refcounting **2× slower than Boehm**.
  **Composability check**: five aspects × two choices = 32 possible
  translations; the predicted cost of stackless+thunk (1.06 × 1.08 ≈ 1.14×)
  matched the measured 1.15×.

### 3.7 Memory management (D05.3 §5, D07.1 §4.2)

- **Boehm** (conservative) and **naive refcounting** were the first two
  production GCs. The Boehm transformer "does mostly nothing but replace the
  allocating operations with the special Boehm versions" — plus it exploits
  toolchain knowledge to pick the best allocation function per type
  (pointer-free objects like strings use the untraced variant). Documented
  Boehm pitfall: conservative scanning kept objects alive whenever an
  integer had a pointer's bit pattern — which *actually happened* because
  identity hashes were the memory address; the fix was to use the **bit-wise
  inverse** of the address as hash. Measured: Boehm 4.4–5.8× slower than
  CPython; "Boehm seems to be the fastest garbage collector option we
  currently have."
- **Refcounting** is judged in its own report: placement "far from being
  optimal" (no redundancy elimination, no trusted references), no cycle
  collection at all ("a fundamental flaw of reference counting … CPython
  solves this with special cyclic-garbage code that PyPy lacks"), 7.7–7.8×
  slower than CPython — "one of the slowest of our GCs"; "not … a very
  interesting area to work on since it is a lot easier to get high
  performance using other algorithms," though nominally "a possibly viable
  option" pending optimization.
- **The GC framework**: exact GCs written in RPython against `Address`
  objects, testable on a **memory simulator** on CPython ("does not involve
  constant segmentation faults"), then woven in by the **GC transformer**.
  Explicitly inspired by MMTk ("Much more work and fine-tuning has gone into
  MMTK, though"). Details the reports record:
  - The transformer replaces allocations with GC helper calls and
    reads/writes with **read/write-barrier calls** where the GC requires
    them; performance-critical helpers (allocation fast path, write barrier)
    can be marked always-inline — "similar to the control over inlining that
    MMTK uses."
  - Object layout information flows through **compile-time `typeid`
    tables** (all types known statically).
  - The simulator evolved from byte-level simulated memory to symbolic
    address arithmetic: address offsets "are represented purely
    symbolically," arenas are simulated objects that can be declared
    exhausted for testing, and freed explicitly-managed objects give useful
    errors when touched. GC-internal data structures are flagged for
    explicit management.
  - Three framework GCs existed at D05.3 (copying, mark-and-sweep, deferred
    refcounting) but ran only on the simulator; by D07.1 **mark-and-sweep**
    was the production framework GC: two header words per object (mark bit +
    typeid; a link chaining all allocated objects), finalizer-needing
    objects on a separate list, resurrection supported, and finalizers run
    **at most once** — a deliberate behaviour change from CPython, where
    finalizers can run "arbitrarily often — a fact that has been a source
    for interpreter crashes." Measured: 5.7–5.8× slower than CPython
    ("quite reasonable … although not quite as fast as with the Boehm
    collector").
- **Root finding** is called "one of the hardest problems" without compiler
  cooperation — "especially hard … using only ANSI C and in a platform
  independent way." The shipped solution is an explicit **root stack**
  (push/pop inserted by the transformer; "not overly good performance").
  The implemented alternative — unwind the stack via the stackless
  transformation and read roots from the heap frame chain — was **measured
  worse**: 13–15% slower and +70% code (every malloc becomes a potential
  unwind point), "very bad instruction-cache behaviour neutralizing
  improvements." A stated future possibility: "use information from the
  still-unfinished just-in-time compiler to find roots on the C stack" (the
  JIT knows the frame layout it generated).
- **Moving/copying GCs** were future work, blocked at the time by two named
  obstacles: consistent identity hashes across moves (the PBC hash-field
  technique of §3.4 is noted as the required shape) and **rctypes** (the
  external-call machinery could not tolerate moving objects — "the major
  factor stopping us from experimenting with advanced GCs so far," per
  D03.1).
- D05.3's **escape-analysis** precursor ("exploding" an object into one
  variable per field when it provably dies with its frame) is recorded with
  the caveat that it "only works well in the presence of function inlining"
  since most allocations happen inside small helpers.

### 3.8 Stackless (D05.3 §6.3, D05.4 §4.1, D07.1 §3)

Motto: "A Python Implementation That Uses the C Stack as a Cache."

- **The design decision underneath**: the interpreter deliberately keeps
  **interpreter-level recursion for app-level calls** — in Python almost any
  operation can lead to a non-tail-recursive call, so a non-recursive
  interpreter would be "extremely tedious" (the recorded lesson of Stackless
  Python v1, which "required extensive modifications to support continuing
  past anything other than a direct Python to Python call, which made
  tracking the development of the Python language extremely difficult").
  Instead, the **stackless transformation** rewrites the graphs at
  translation time.
- **Mechanism**: a special `UnwindException` unwinds the whole C stack like
  an uncaught exception, but every interrupted function catches it, saves
  its live locals into a heap frame, and re-raises; the heap mirror is a
  linked list of GC'd frame structures, each holding a back pointer, a
  compact integer identifying function + resume position (one global table
  maps it back), and the live locals. Because each resume point would need
  its own struct layout, the transformer **type-erases and sorts the saved
  fields** "to reduce the number of different layout variants to reasonable
  levels." At the bottom sits a hand-written **driver** that pops and
  resumes saved frames one at a time, transporting return values and
  exceptions between callee and caller.
- **Resume points are placed *after* calls.** The rejected alternative
  (placing them before calls and re-performing the calls — tried in the
  literature, [PICOIL]) is contrasted with two stated advantages of
  after-call placement: it handles frame chains deeper than the C stack
  limit (the whole chain never needs to be resident), and it makes
  microthread switching **amortized constant-time** ("particularly important
  for our main use case of many massively parallel microthreads frequently
  switching control"). Only the innermost frame is ever restored; "the
  'real' stack … is but a cache for the top section of the current
  heap-based frame chain."
- **Stack-overflow checks** are placed by computing **back edges of the call
  graph** and inserting a check before those calls only — every recursive
  cycle meets at least one check, fast paths meet none.
- **Performance design goal**: "we never take a pointer to any local
  variable," so C register allocation still works; restore writes happen
  immediately before a jump to the resume point, so they compile to writes
  into the allocated registers. Measured (rev 34906): +17–19% time on
  Boehm builds, +26–28% on framework-GC builds; code 2.1–2.4×. The rejected
  "locals in a struct" approach measured **+60–64%**. D05.4's earlier
  numbers (+8%, +28% size) predate the 0.9-era rework. Stackless features
  "are not thread-safe so far … although this is not a deep problem."
- **The primitive layer**: `stack_unwind()`, `stack_frames_depth()`, and the
  core coroutine primitive `yield_current_frame_to_caller()` returning an
  opaque `frame_stack_top` with `.switch()`. Documented hard rules: every
  `frame_stack_top` must be resumed **exactly once** (never → leak; twice →
  crash), and a function that yielded its frame must not raise (no implicit
  parent). The interface is "extremely primitive" by design and wrapped in
  plain RPython (the Coroutine class); on more flexible platforms the same
  interface could be implemented natively (Squeak, or assembly stack
  copying).
- **Stack reconstruction** — named resume points (`resume_point("label",
  vars…, returns=varR)`), `resume_state_create(back_frame, "label",
  values…)` and `resume_state_invoke(return_type, frame_top)` let an RPython
  program "artificially rebuild a chain of calls in a reflective way,
  completely from scratch, and jump to it." "We know of no equivalent of
  this feature in the literature." The consumer is coroutine pickling:
  unpickling a chain of Python-level frames requires manufacturing the
  corresponding interpreter-level frame chain.
- **App-level offering** (the 0.9 `stackless` module — interface declared
  "in-flux"): **coroutines** (explicit switch; bind/switch/kill; kill's
  exception "currently not visible at app-level, so uncatchable and
  try/finally not honored — will be fixed in the future"); **tasklets and
  channels** ("roughly compatible" with Stackless Python, implemented *at
  application level* in `pypy/lib/stackless.py` on top of coroutines — "the
  code tries to resemble the stackless C code as much as possible. This
  makes the code somewhat unpythonic"; strictly round-robin scheduling;
  channels are meeting points, not queues); **greenlets** (tree-structured,
  identical to the py-lib interface; PyPy's "do not suffer from the cyclic
  GC limitation that the CPython greenlets have").
- **Coroutine pickling**: frames are fully picklable (bytecode reference +
  locals); globals/modules pickled by name; the pickle stores a bytecode
  index, so "the user program cannot be modified at all between pickling and
  unpickling!"; pickling **fails if the suspension point involves indirectly
  invoked Python frames** (operators calling `__xyz__`, builtins calling
  back, signal handlers) — "not supported yet." Measured: pickling ~7–9 ms,
  unpickling ~10–12 ms ("implemented mostly at application-level, which
  explains their slowness").
- **Coroutine cloning** (GC-pool based; the claimed novel contribution:
  "The approach of using the GC to allow a form of coroutine cloning is, to
  the best of our knowledge, novel"): correct semantics chosen = duplicate
  exactly the objects *created by* the cloned coroutine, share the rest;
  implemented by giving the GC a per-coroutine allocation **pool** and
  switching pools on coroutine switch; byte-level copies with recursive
  copying inside the pool. Requires `--stackless --gc=framework` together.
  `fork()` returns a child clone UNIX-style. The showcased consumer is
  logic-programming search (§7). The report's conclusion holds this up as
  the integration proof: cloning uses the GC pools + the stackless heap
  frames + the coroutine abstraction — "no C code was involved in any of
  this."
- **Composability**: coroutine models "do not compose naturally" — two
  unrelated usages of switching conflict through the single implicit "main."
  The fix: per-user **"views"** (`stackless.usercostate`), each with its own
  main/current; switching is implicit within the target's view. Stated
  future direction: "we will probably change the app-level interface … to
  not expose coroutines and greenlets at all, but only views." The
  composability design is claimed as new "as far as we know."
- **Microbenchmarks** (D07.1 §3.3.7): PyPy coroutine/greenlet switch time is
  **independent of stack depth** (~6–17 μs), while CPython's greenlet switch
  is **linear in depth** (1.81 μs at depth 0 → 25.93 μs at depth 100 — each
  switch memcpys the whole greenlet stack). OS threads: 8.4 MB virtual per
  thread on default Linux → a 32-bit address space exhausts at ~382 threads.
- **Motivating deployment**: CCP Games (EVE Online) used and sponsored
  Stackless Python; a CEO quote credits its concurrency model as the
  foundation of their success. The stated application classes: simulations
  with thousands of agents; servers with many concurrent clients; web
  servers resuming continuations ("breaking the back button" example);
  checkpoint/migration in clusters; and logic-programming choice points via
  cloning.

---
## 4. Core optimizations (D06.1) and translation-level optimizations (D07.1 §4.1)

### 4.1 Method

"Because optimization is at its core such an empirical activity, we begin by
describing the process by which we assessed the benefit of each attempt at
optimization." Five benchmarks were settled on (finding suitable ones is
called "no trivial task"):

| Benchmark | What it stresses (as stated) |
|---|---|
| pystone | "the archetypal Python benchmark, but in no way typical Python code" (Dhrystone lineage) |
| richards | "of all the benchmarks most tests the performance of method calls" |
| templess | regular expressions and string handling |
| gadfly | pure-Python SQL database; non-trivial algorithms, heavy long-integer use |
| mako | Unicode handling |

The last three allocate much more memory, hence stress the GC. Plus
microbenchmarks per interpreter part, and **automated nightly runs** of
pystone/richards to catch regressions. Environment: a 4-core Xeon 3.2 GHz
(single core used). Headline: "the speed of a compiled PyPy with all
optimizations enabled is more than twice that of a binary with some
optimizations disabled" — and the real total is higher, since clear wins
adopted before the D13.1 toggle framework were never made optional.

### 4.2 Multimethod dispatch compression

- The original automatically-generated **chained double dispatch** grew
  quadratically: N types × N vtable entries per doubly-dispatched
  multimethod, until vtables occupied "more than 50% of the total size" of
  the executable.
- Replaced by **Multiple Row Displacement** (Pang et al.): a pair of
  integers per vtable plus small compressed tables. Effect on the C
  backend: **−2.5 MB, almost half the binary**, at "a few percent slower"
  runtime — explained as double-dispatch being one double indirect jump vs
  MRD's ~4 arithmetic instructions + one indirect jump, "and indirect jumps
  are costly on modern processors, so effects even out." Verdict:
  "double-dispatch is marginally faster … but requires a prohibitively large
  amount of static data, making [MRD] a better practical choice."
- On the OO backend (pypy-cli/Mono) the trade inverts: **MRD measured 15%
  slower**, so double dispatch remained the OO default. (Indirect *method*
  calls are what OO platforms optimize; indirect *function* calls need
  delegates.)

### 4.3 The D06.1 empirical record, optimization by optimization

Author-judged results, with the numbers they reported (baseline 1.0, lower
is better, unless stated):

| Optimization | Verdict / numbers |
|---|---|
| Prebuilt small-integer boxes (−5..99 / −5..256) | **Mixed/negative**: 0.98–1.06 across benchmarks; suspected **bad cache effects** (fresh integers are cache-hot; prebuilt ones need extra loads); the marginal win required "a hack to prefetch the `intval` field" |
| String join objects, string slice objects, ropes | 3% faster to 9% slower, "no clear tendency" — apps are tuned for flat strings; benefits only for code written against the new complexity profile (e.g. an editor buffer over ropes); noted side-benefit: ropes keep naive algorithms efficient |
| Multidicts + StrDictImplementation | **15–40% improvement**, "the bulk likely from not needing method dispatch to check key equality"; "dictionaries play a central role in Python, performance-wise" |
| SmallStr/SmallDict (sequential-search small dicts) | "almost no effect" — most lookups hit a few large dicts (modules, builtins) |
| SharedDict (key-sharing instance dicts) | Memory for 1M two-attribute instances: 140,136 KB vs 266,860 KB baseline (CPython 202,548; `__slots__` 96,640) — "`__slots__` behaviour, transparently"; speed ~1% faster to 10% slower |
| Multilists: range lists, list slices, chunk lists | 3% faster to 11% slower; noted: with range lists "xrange would not be necessary" |
| Optimal resizable arrays (Brodnik) | "as bad as 4 times slower" — optimal in theory, huge constants |
| CALL_LIKELY_BUILTIN + shadowing-tracking module dicts | 0.81–0.89 (11–19% faster); "not a pure win — it slows writes to global variables slightly" |
| Shadow tracking (instance-attr shadowing flag on dict impl) | richards 21%, pystone 14%, templess 4% faster; disabled entirely for instances that ever shadow — "further research could improve on this" |
| Method cache (per-class version_tag, global (tag,name) table) | richards up to 23% at 1024 entries; 128 entries already good, 512 sufficient for templess; "caches the result of *any* lookup"; compared with PICs: one global cache uses less memory and is "unlikely to miss often in tight loops" |
| LOOKUP_METHOD/CALL_METHOD | ~6%; "either a net win or have no effect" |
| All three method optimizations combined | richards 0.53, pystone 0.78, templess 0.75 — "up to 47% … surpassing even what could have been expected from the individual speed-ups" |
| Precomputed `__xyz__` lookups on builtin types (translation-time, `cache__xyz__` fields on W_TypeObject) | "an obvious big win with no trade-off": disabling costs 22–32%; included in every baseline |
| Switch-based dispatch loop | Mixed 0.91–1.05; "we can only guess" — suspected code-size/locality effects after inlining |
| Argument-passing fast paths | call microbenchmarks: 5.03–6.56× slower than CPython (Mar 2006) → **2.65–3.44×** (Mar 2007) |
| Type-special-casing in bytecodes (int+int, list[int]) | 0.96–1.02, "did not give conclusive speed-ups … the normal dispatching mechanisms are well-optimized already" |
| String interning; app-level-string-accepting interfaces; exception-avoiding interfaces | reported without numbers; pervasive |

Mechanism notes the reports give for the load-bearing ones:

- **Multidicts**: a dict is a thin identity wrapper delegating to a swappable
  implementation object; mutators may return a replacement implementation.
  Ten implementations existed (Empty, SmallStr, Str, Small, RDict, Bucket,
  Shared, Wary, ShadowTracking, Measuring). Empty is a singleton on the
  space (fresh dicts are cheap); the first insert picks Str vs RDict by key
  type; switching later "is an exceedingly rare occurrence," which justified
  not sharing storage across implementations. MeasuringDictImplementation
  existed purely to collect usage statistics (analysed with R).
  WaryDict/ShadowTracking serve CALL_LIKELY_BUILTIN and shadow tracking.
- **CALL_LIKELY_BUILTIN**: the compiler emits it for calls whose target name
  is a builtin; a special module-dict implementation tracks which builtin
  names are shadowed by globals; the bytecode makes two cheap checks
  (module-dict shadowing? `__builtin__` changed?) and on the common
  both-false path calls the builtin directly with no dict lookups.
- **Method cache**: `version_tag` is a tiny object on each type; any class-
  dict change replaces the tag on the class *and all subclasses*, so the
  global (tag, name) table can never return stale entries. Rejected
  alternative recorded: per-class "flattened" namespace dicts (fast but
  non-linear memory for deep hierarchies). Future hardening listed: 2/4-way
  set-associative tables, or randomly refreshing a type's tag after
  repeated misses.
- **CALL_METHOD**: `obj.meth(x)` normally materializes a bound-method object
  at LOAD_ATTR that dies immediately. LOOKUP_METHOD performs the same lookup
  with full semantics but pushes an *inlined* bound method as two stack
  slots (im_func, im_self — None placeholder when the attribute wasn't a
  plain class function); CALL_METHOD N inspects the im_self slot and, when
  set, treats it as an extra first argument. "A few variants were tried;
  this one is the first that fully preserves semantics and gives major
  speed-ups." CPython patches in this space ([py709744] CALL_ATTR) had
  stalled on complexity/benefit balance; PyPy's method cache "already has
  been" ported to CPython ([py1685986]).
- **Dispatch loop**: Python has no switch statement, so dispatch was written
  as an if-chain made viable by a translation-time pass that recognizes
  chains of comparisons of one value against constants and emits a switch
  in the flow graph.
- **Avoiding exceptions**: performance-critical interfaces return None
  instead of raising (`getdictvalue`, `finditem`) because exceptions "must
  be instantiated → memory pressure."

### 4.4 Bottom line of D06.1

All-toggleable-optimizations-on vs off: richards 0.40, pystone 0.56,
templess 0.65, gadfly 0.71, mako 0.62. Against CPython 2.4.4 (best backend,
pypy-llvm): richards 1.17×, pystone 1.55×, templess 5.41×, gadfly 6.38×,
mako 7.65× slower — "depends very strongly on the amount of memory being
allocated"; "the biggest single improvement would likely be from the
addition of a more sophisticated garbage collector, something that is out of
scope for this work package." Closing claim: "The flexibility of PyPy's
Python implementation has been essential to the task of experimenting with
and especially assessing these new implementations, vindicating the effort
we put in to allow this flexibility."

### 4.5 Translation-level optimizations (D07.1 §4.1)

Context the report gives: on first full translation, pypy-c was "between 22
times slower (for the Richards benchmark) and roughly 460 times slower for
the pystone benchmark" than CPython; "one of the biggest sources of
inefficiencies is the large amount of time spent managing memory" — RPython
allocates at an extremely fast rate, so most optimizations reduce
allocations or predict lifetimes.

- **Inlining**: copy the callee graph into the caller; statically-known
  targets only. "Primary motivation: to enable further optimizations
  (notably malloc removal)." The explicit-exception-edge graph model makes
  inlining across try/except hard (each potentially-raising operation gets
  explicit handler jumps; a heuristic matches directly-caught exceptions and
  "in more complicated cases inlining is not performed"). Heuristics: static
  instruction count + a "median execution cost" solving linear equations
  over guessed branch frequencies; smallest-first with recalculation; ties
  made **deterministic** (fewest-callers-first) after nondeterminism proved
  troublesome. Alone: pystone +52%, richards +64% — "these numbers probably
  show that … the GCC compiler itself could benefit from performing more
  aggressive inlining."
- **Malloc removal**: `for i in range(n)` allocates three heap objects (the
  range, its iterator, the StopIteration instance) that this kills after
  inlining. Candidate = a variable set created by a single malloc and only
  accessed by field reads/writes; the analysis gives up on escape into
  calls/returns/other structures — unless the containing structure is itself
  removed, which can re-enable the candidate. Removes ~20% of mallocs on the
  default build; on top of inlining: richards +18%, pystone +43%. "Made
  useful only by the inliner … inlining was tweaked in such a way that
  malloc removal can work effectively."
- **Escape analysis / stack allocation**: deliberately "naive, pessimistic"
  (creation-point sets, forward-propagated; escaping = returned, raised, or
  stored into another object; stack allocation refused for variable-sized
  objects, allocations in loops, and finalizable objects). With malloc
  removal on: only +3–7% (they cover the same cases); with malloc removal
  off: +40% pystone / +30% richards.
- **Pointer tagging**: implemented "uncharacteristically … not by pervasive
  source code changes, but by minimal support in the translation tool-chain"
  — an optional **156-line module**. Small integers are a regular RPython
  class in the source; during translation instances become odd-tagged words;
  only the "read dynamic type" operation was changed (tag bit → the small-int
  class vtable). It pays off only combined with **record-time constant
  folding of vtable reads** (the generated vtables are constant structures;
  "C compilers can't do this — they lack this info"), which turns the
  indirect method call into a direct one. Measured: without const-folding
  pystone +4% / richards −3%; with const-folding pystone +7% / richards +1%.
  Fundamentally incompatible with CLI/JVM backends; at 1.0 only supported
  with Boehm. Claimed novelty: "the automatic insertion of pointer tagging
  as a translation-time optimization is novel in PyPy."

The D07.1 consolidated table (July 2006 revision, AMD Opteron; relative to
the default build = inlining + malloc removal + Boehm): no-optimizations
2.20–2.24×; just-inlining 1.34–1.47×; inlining+malloc-removal 1.03–1.13×;
escape-analysis ~0.93–0.97; tagged-pointers+constfold 0.93–0.99; refcounting
1.77–1.79; mark-and-sweep 1.30; mark-and-sweep with stackless-unwind roots
1.55–1.63. CPython on the same machine: 0.23. (pypy-c-0.7, refcounting:
104.7× on pystone — the starting point of the campaign.)

A recurring stated lesson closes the section: theoretically superior data
structures usually show no benefit on real applications tuned for CPython's
cost model; the flexibility to *measure* alternatives on real programs is
presented as the valuable thing.

---
## 5. The 2007 JIT: partial evaluation and the JIT generator (D08.1, D08.2)

**Scope note (from the documents themselves)**: this is the first-generation
JIT — a *compiler generator* derived by binding-time analysis and
"timeshifting" of the interpreter's low-level graphs. The documents do not
mention tracing.

### 5.1 Goal and framing

- "The main research goal … since the inception of the PyPy project": don't
  write a JIT; **generate** it from the interpreter. Hand-written JITs "may
  require a lot of effort" and "may well be fragile with respect to changes
  to language and its semantics"; open-source language communities "do not
  usually consider ease of compilation as a design constraint"; the typical
  main implementation is "a straight-forward bytecode interpreter with no
  dynamic compilation."
- The framework is **off-line partial evaluation**: for an interpreter
  `f(x, y)` (x = the bytecode, y = the frame), produce a *generating
  extension* `f1(x)` — "a compiler for the very same language for which
  f(x, y) is an interpreter." Footnote: "What we get in PyPy is more
  precisely a just-in-time compiler: if promotion is used, compiling ahead
  of time is not possible."
- **Rejected alternative, explicitly**: building the generating extension by
  self-applying an on-line partial evaluator (the second Futamura
  projection) — "So far it is unclear if this approach can lead to optimal
  results, or even if it scales well. In PyPy we selected a more direct
  approach: the generating extension is produced by transformation of the
  control flow graphs of the interpreter, guided by the binding times. We
  call this process timeshifting."
- **Classical PE rejected as insufficient** for dynamic languages: "the
  input program contains mostly no kind of type information"; therefore
  "Compilation should be able to suspend, let the produced code run to
  collect run-time information (for example language-level types), and then
  resume with this extra information."
- Three phases are distinguished: **translation time** (purely off-line:
  binding-time analysis + timeshifting), **compile time** (the generated
  compiler running during program execution), **run time** — "compile time
  and run time are actually highly interleaved."
- The three techniques named as what made the approach scale: **promotion**,
  **virtualizable structures**, and **need-oriented binding-time analysis**.

### 5.2 Machinery

**Binding-time analysis.** Performed on the interpreter's *low-level* graphs
(post-RTyping) by the annotator re-used in a new mode — the
**hint-annotator** — propagating "not types but value dependencies and
manually-provided binding time hints." **Green** = compile-time-known,
**red** = runtime; "all variables are red by default." Propagation: a
side-effect-free operation with all-green arguments may be green;
non-foldable operations are red; at join points any red input makes the
result red (a green join is *not* made eagerly, because the residual
function may retain control flow the join value depends on).

**Hints are need-oriented.** The design goal is to "minimize the number of
explicit hints" without pushing to extremes. `hint(v, concrete=True)`
requires the value green and **backward-forces everything it depends on to
be green**, erroring if a dependency cannot be green (e.g. a read from a
non-immutable field). The stated rationale against forward propagation: it
"may mark as compile-time either more variables than expected (which leads
to over-specialization …) or less variables than expected (preventing
specialization … where it would be the most useful)"; the need-oriented
approach "reduces the problem of over-specialization, and it prevents
under-specialization: an unsatisfiable hint … is reported as an error,"
correctable "by promoting a well-chosen variable among the ones that v1
depends on." `hint(v, promote=True)` is the local escape hatch: "copying a
red value into a green one, which is not possible in classical approaches to
partial evaluation." Other hints: **deep-freeze** (mark an object immutable
so reads constant-fold), **global merge points**, `_virtualizable_`.

**Timeshifting.** The colored graphs are mutated into the generating
extension — "it changes the time at which the graphs are meant to be run,
from run-time to compile-time." Green-only pure operations stay unchanged
and run at compile time — "(This is the case that makes the whole approach
worthwhile: some operations become purely compile-time.)" Red operations
become calls to helpers that emit a residual operation and return a **boxed
representation** of the result — a box holding either a runtime location
(register/stack slot) or an immediate constant. Because boxes can hold
immediate constants, the helpers constant-fold — "the timeshifted graphs are
performing some **on-line partial evaluation** in addition to the off-line
job." Timeshifting runs in two phases: graph transformation inserting
pseudo-operations, then an RTyper-based pass replacing them with support-code
calls.

**Splits and merges.** A compile-time-undecided branch duplicates the
compilation state and compiles both sides. At merge points the state must be
generalized (widened) or paths kept separate; the classical tension is
quoted — "merging too eagerly may loose important precision and not merging
eagerly enough may create too many redundant residual code paths (to the
point of preventing termination of the compiler)" — and the resolution is
admitted to be provisional: "So far, we did not investigate this problem in
detail. We settled for a simple widening heuristic: two different
compile-time constants merge as a run-time value … This heuristic seems to
work for PyPy to some extent."

**Calls.** Timeshifted callees are inlined into the caller by default;
"inlining only stops at re-entrant calls to the interpreter main loop," so
"at the level of the interpreted language, each function (or method) gets
compiled into a single piece of residual code."

**Promotion.** "The essential new feature introduced in PyPy when compared
to existing partial evaluation techniques (it was actually first introduced
in Psyco)"; the converse of PE's "lift"; "the central enabling primitive to
make timeshifting a practical approach to language independent dynamic
compiler generation." The documents' framing points:

- Why it is impossible classically: "Promotion requires interleaving
  compile-time and run-time phases … impossible in the 'classical'
  approaches … in which the compiler always runs fully ahead of execution."
- The constraint-theoretic argument: binding times are mutually constrained
  and for a dynamic-language interpreter "this set of constraints may have
  no interesting global solution" — most variables can, in some corner case,
  depend on runtime data. Promotion lets constraints "be occasionally
  violated: corner cases do not necessarily have to influence the common
  case, and local solutions can be patched together."
- It generalizes **polymorphic inline caches** (the generated updatable
  switch "plays the role of a polymorphic inline cache," except the switch
  "does not necessarily have to be on the type of an object") and PE's
  "The Trick" (which "is only applicable for finite sets of values").
- Implementation: the generated code contains an updatable switch ending in
  a `continue_compilation(value, <state data pointer>)` callback; on a new
  runtime value the compiler resumes, emits the new case, and **patches the
  switch in place**. The worked example in D08.2 shows `b = a/10;
  c = promote(b); d = c + 5` compiling to a compare-chain that grows a
  `call print(9)` case after first seeing `r2 == 4`.
- **State compression ("paths")**: a full compiler-state snapshot per switch
  "can never be reclaimed because new run-time values may always show up
  later"; instead full snapshots are stored only at user-marked **global
  merge points** (the interpreter main-loop join "is a typical place"), and
  each promotion stores only a lightweight path in a tree that lets the
  compiler *replay* its actions from the last snapshot to rebuild its state.
- **Space exhaustion is handled crudely (their word)**: "we simply reserve
  more space elsewhere and patch the final jump accordingly"; better
  strategies (discarding old cases, "sometimes giving up entirely and
  compiling a general version") are listed as unimplemented.
- Guidance on where to promote (Appendix 1): small integers; "the
  interpreter-level class of an object before an indirect method call" (a
  constant class lets the compiler inline the callee); occasionally, "with
  care," whole objects (e.g. exactly which Function object is being called).

**Virtual structures.** Allocations are kept "exploded" — the compiler
variable holds a *virtual structure* containing one variable per field, each
of which can be runtime, compile-time, or again virtual — until the pointer
**escapes** to a non-virtual location ("forcing"). Because red values
represent *locations*, a getfield/setfield on a virtual structure just moves
location references at compile time; nothing is copied at run time. The
worked example: in `a+b+c` over `W_IntObject`, the intermediate box becomes
virtual, its `intval` referencing the register holding the first addition's
result, and the promotion of its (compile-time-constant) type is free —
"a+b+c only requires three switches instead of four." Lists and
dictionaries can also be virtual; the **exception state is handled by
explicit operations inserted before timeshifting and then virtualized
away**. In a tight integer loop "the residual code is theoretically optimal
— all type propagation and boxing/unboxing occurs at compile-time."

**Virtualizable structures.** Invented for **frame objects**, which cannot
be virtual because they are sometimes built by non-JIT code and in any case
"immediately escape into the global list of frames that is used to support
the frame stack introspection primitives that Python exposes" — "even though
in practice most of frame objects are deallocated without ever having been
introspected." A virtualizable structure "exists at run-time in the heap,
but is simultaneously treated as virtual by the compiler." The class-level
`_virtualizable_` hint makes the toolchain add a **hidden field** to such
objects; every field access anywhere in the program first checks it, and
non-JIT code finding it set "invokes a JIT-generated callback to perform the
reading or updating of the field from the point of view of its virtual
structure representation." "This is the only case so far in which the
presence of the JIT compiler imposes a global change to the rest of the
program." Frames stay heap-allocated "but most of them will always remain
essentially empty," and introspection "still work[s] perfectly." The
indirection cost is acknowledged, with a mitigation: a declaration usable
during type inference that a given code region cannot see a
virtual-counterpart frame, eliding the check.

**Correctness by construction.** "By construction, the JIT should work
correctly on absolutely any kind of Python code: generators, nested scopes,
exec statements, `sys._getframe().f_back.f_back.f_locals`, etc." — the
latter "an example of expression that no existing Python or Python-like
compiler emulates correctly." The contrast drawn with Psyco: Psyco "gives up
compiling Python functions if they use constructs it does not support, and
is not 100% compatible with introspection of frames. By construction the
PyPy JIT does not have these limitations."

**Code size.** Support code (merge logic etc.): ~3,500 lines RPython;
hint-annotator + timeshifter: ~3,800 lines Python; each machine backend
(IA32, PPC): ~3,500 lines RPython. "There is a well-defined interface
between the JIT compiler support code and the backends … The unusual part of
the interface is the support for the run-time updatable switches."

### 5.3 Operation as shipped in 1.0 (D08.1)

- Build: `translate.py --jit targetpypystandalone`. The resulting pypy-c
  contains both a regular interpreter and the generated compiler — and the
  built-in JIT makes the *interpreter itself* "a bit slower than the one
  found in a pypy-c compiled without JIT."
- **Invocation is manual**: `import pypyjit; pypyjit.enable(f.func_code)`;
  the first call compiles, subsequent calls run machine code. "We did not
  work yet on profile-directed identification of program hot spots."
- Entry mechanism (Appendix 1, tiny1 example): the timeshifted function is
  patched so that a call looks up its **green arguments** (e.g. the bytecode
  string) in a cache; on a miss it invokes the compiler with the greens as
  constants and the reds as variables; then it invokes the cached machine
  code. "Interpreting the same bytecode over and over again with different
  values of x and y should be the fast path."
- Backends: IA32 and PowerPC; tested on Linux, Mac OS X (Intel and PPC),
  Windows. Machine code dumps via `PYPYJITLOG` + a viewer built on objdump.
- Configuration: the JIT was supported **only in the default configuration**
  (D13.1 matrix, §6.4); combining with thunk/taint spaces "probably works
  too, but we don't expect it to generate good code before we add some extra
  hints in the source code of the object spaces."
- Testing admission: "Although the JIT generation process is well-tested, we
  only have a few tests directly for the final pypy-c."
- The framing quote of the appendix: "Ideally, turning an interpreter into a
  JIT compiler is only a matter of adding a few hints. In practice, the
  current JIT generation framework has many limitations and rough edges
  requiring workarounds."

### 5.4 Results and admitted limitations

- Benchmark: one pure-arithmetic nested-while function, f1(2117). Reference
  machine numbers (ratios vs unoptimized gcc; "relative results have been
  found to vary by 25% depending on the machine"):

  | Configuration | sec/call | vs unopt. gcc |
  |---|---|---|
  | CPython 2.4.4 | 0.82 | 132× |
  | CPython + Psyco 1.5.2 | 0.0062 | 1.00× |
  | pypy-c, JIT off | 1.77 | 285× |
  | pypy-c, JIT on | 0.0091 | **1.47×** |
  | gcc | 0.0062 | 1× |
  | gcc -O2 | 0.0022 | 0.35× |

  "Matches the target of 1.5× that we set ourselves"; 1.15× on one Intel
  Mac. Interpretation offered: "all the abstraction overhead has been
  correctly removed"; the remaining gap is "only due to a suboptimal
  low-level machine code generation backend." The result "require[s] the
  generated compiler to completely cut the overhead and fold at compile-time
  some rather involved lookup algorithms like Python's binary operation
  dispatch. Promotion proved itself to be sufficiently powerful to achieve
  this."
- Coverage honestly scoped: "enough hints were added … such that at least
  integer operations can be dynamically compiled into efficient code and
  dispatch loop overhead is removed. … more extensive hints are necessary to
  have more generalized speed-ups. Their addition is going to be part of
  work to be done after the project."
- Interpreter-side cost claimed small: "Some slight reorganisation of the
  interpreter main loop without semantics influence, marking the frames as
  virtualizable, and adding hints at a few crucial points was all that was
  necessary."
- **Open issues, in the authors' own list (D08.2 §3.7)**:
  - Eager branch compilation "can easily result in residual code explosion.
    Depending on the source interpreter this can also result in
    non-termination issues, where compilation never completes." Psyco-style
    fully-lazy compilation "neatly sidesteps termination issues"; "the best
    solution is probably something in between these extremes."
  - Fallbacks needed when too many values are promoted at one point;
    "the widening heuristics for merging needs to be refined"; "we need more
    flexible control about what to inline."
  - The JIT must be made aware of the other translation aspects (GC,
    stackless) to emit correct residual code.
  - "The machine code backends can be improved."
  - The stated **open research question**: "can we layer our kind of JIT
    compiler on top of a virtual machine that already contains a lower-level
    JIT compiler? … can we delegate the difficult questions of machine code
    generation to a lower independent layer, e.g. inlining, re-optimization
    …?"
- Closing claim (D08.2 §5): "our results make viable an approach to
  implement dynamic languages that needs only a straight-forward bytecode
  interpreter to be written. Dynamic compilers would be generated
  automatically guided by the placement of hints" — implementations "robust
  against language changes up to the need to maintain and possibly change
  the hints."

---
## 6. Backends, extension modules, configuration (D03.1, D12.1, D13.1)

### 6.1 Language tracking and the extension compiler (D03.1)

**Language tracking.** PyPy started on CPython 2.3.x, switched to **2.4.1**
in summer 2005 and deliberately held there — 2.5 "is not yet widely
deployed, so that programmers still restrain from using the new features";
"it is customary for a Python version to only become widely accepted and
relied upon after the '.1' or '.2' release." Selected 2.5 features were
backported (conditional expressions PEP 308, `with` PEP 343); others were
"clean-ups of historical accidents" PyPy already had (new-style exceptions,
ssize_t sizes); `__index__` (PEP 357) was "yet to be implemented." Adopting
features was "fairly easy" thanks to "the expressiveness of Python compared
to C and the flexibility of relevant parts of the design (e.g. our bytecode
compiler)." No automated CPython-tracking tooling was built — judged
"superfluous" because many PyPy developers were themselves active CPython
contributors.

**Extension-module strategy.** The stated core drawback of C extensions for
PyPy: explicit low-level detail (locks, memory management) "would prevent
the same module from being used in several differently compiled versions of
pypy-c." The chosen approach: write extension modules in **RPython**, as
**mixed modules**, usable **four ways** from one source — (1) directly on
CPython through a **CPy Object Space** (whose space operations are
"essentially just external function calls to the C functions of the CPython
API"), (2) compiled by the stand-alone **Extension Compiler** into a regular
CPython extension `.so`, (3) under PyPy-on-CPython, (4) built into pypy-c.
"The translation step is optional: it is a way to recover performance"
(modules are plain ctypes-using Python until translated) — the stated
contrast with Pyrex, whose input is not actually Python. The one-source
approach "could be adapted to target other Python implementations as well
(Jython, IronPython), enabling a one-source-fits-all approach."

**rctypes** (ctypes restricted to RPython, statically compiled): dynamic
ctypes declarations are allowed at bootstrap only; during translation "all
replaced by regular, static C code that no longer uses the libffi library."
The report documents an explicit **mid-flight pivot** between two
implementations:

1. **First approach — RTyper-integrated**: each ctypes operation lowered
   individually during RTyping; memory-owning boxes mapped to
   `GcStruct("name_owner", ("c_data", Struct(...)))` where the inner struct
   follows the exact C layout. The recorded reasons for abandoning it:
   "quite tedious to have to write code that generates detailed low-level
   operations for each high-level ctypes operation — and most importantly,
   it is not flexible enough"; "problems were recently found in corner cases
   of memory management"; and **it blocked moving GCs** — "the major factor
   stopping us from experimenting with advanced GCs so far."
2. **Second approach — a pure-RPython library (`rctypesobject`) driven by
   the generic ControllerEntry mechanism** (controllers map operations on
   arbitrary Python objects to glue RPython classes during annotation).
   Work in progress at report time; **no performance figures were
   collected**; work "postponed … in favor of more urgent tasks."

Recorded gaps: c_float/c_wchar unsupported; custom allocators and
return-value error checkers unsupported; the extension compiler "does not
support special methods at the moment" (`__xyz__`). The section's own
conclusion: "The Extension Compiler is still work in progress."

### 6.2 High-level backends and ootype (D12.1)

- **ootype** was created because lltype "loses too much information" for OO
  targets (an RPython list lowered to `struct {int length; int* items;}`
  retains no trace of being a list). Its model is "quite Java-like": static
  strongly-typed methods/attributes, all methods virtual, **single
  inheritance** from ROOT ("it's impossible to efficiently support multiple
  inheritance if the platform supports only single inheritance"), built-in
  String/StringBuilder/List/Dict/CustomDict parametrized by type, method
  overloading supported only for native-library access. Both type systems
  share the arithmetic primitives.
- The deliberate contrast with Jython/IronPython: those compile Python to
  *native* bytecode with a runtime in the host language; PyPy compiles the
  *interpreter* to the platform and keeps executing Python bytecode on it —
  "this extra level of indirection results in a speed penalty … but also
  gives a lot more flexibility" (e.g. swappable object spaces). Host-library
  integration "is not completely supported in PyPy yet, but it will be."
- **GenCLI** (the most mature; from Antonio Cuni's master's thesis): emits
  IL for ilasm, "starting from the entry point and recursively traversing
  the call tree — considerably simpler than the approach of GenC."
  Functions-as-values become generated delegate classes; CustomDict becomes
  a generated IEqualityComparer; classes-as-values are `System.Type` +
  reflective `RuntimeNew`. The `clr` module exposes .NET classes at app
  level through three layers (static RPython bindings whose signatures are
  harvested by an external C# reflection tool; a Python/CLI object bridge;
  reflective app-level class construction via a descriptor-based
  MethodWrapper); exceptions "still incomplete": "every .NET exception is
  mapped to Python StandardError."
  **Measured** (richards, ms/iteration): on Windows/MS CLR — C# 7.09,
  gencli-richards 13.31 (≈1.8×), CPython 1139.6, IronPython 1751.2,
  pypy-cli-interp-level 5952.5, pypy-cli-app-level 12010.5. On Linux/Mono
  the GenCLI-vs-C# gap widens to ≈3.4× ("Mono JIT … is specifically
  tailored for code produced by the C# compiler"). The pypy-cli ≈6.8×
  IronPython gap is decomposed as **1.8 (backend) × 2 (interpretation
  overhead) × 1.8 (stdobjspace vs IronPython runtime)**; the backend factor
  is "attributable to the youngness of GenCLI and will likely converge to
  zero."
- **GenJVM**: emits Jasmin assembler ("Sun has not defined a standard
  textual format for JVM byte-code"); "fairly smooth" thanks to ootype and
  reused CLI-backend code, but could not yet translate the full interpreter;
  rpystone ~20× slower than C with "no applied optimizations." Documented
  JVM frictions: no unsigned arithmetic (library comparisons); generics via
  erasure + verified casts; functions as one-class-per-function with virtual
  `invoke`; **exceptions must wrap** (`JvmExceptionWrapper`, losing
  class-based catch — alternatives noted: map RPython Exception to
  Throwable, or signal exceptions via return values); the future hazard of
  the 64 kB method limit (the SSI form is noted as what would make automatic
  method splitting easy).
- **GenJS**: does *not* aim to run the interpreter; compiles RPython
  applications to browser JavaScript with transparently-proxied server calls
  (XMLHttpRequest, auto-serialized given known signatures) and a CPython
  wrapper so the same program is testable off-browser — the stated focus is
  the missing testing story for browser code. Prototype OO is bridged by a
  copy-down `inherit` function, valid because "RPython does not allow
  changing base class members at runtime anyway." External APIs (DOM,
  Mochikit) are described via `BasicExternal` signature stubs — "the real
  methods are never seen by the annotator."

### 6.3 Interpreter feature prototypes (D12.1)

- **Taint object space** (information-flow security): a proxying space
  between interpreter and StdObjSpace, both untouched. Capability-based
  security was explicitly rejected as a direction ("would have mostly
  required working at the language design, and not so much at implementation
  level" — Python has no encapsulation, pervasive reflection, shared
  builtins). Objects are regular or tainted; `taint`/`untaint` builtins;
  every operation with a tainted argument produces a tainted result;
  failures become **tainted bombs** that raise only on declassification
  (immediate errors would leak information); unboxing operations (output,
  truth-testing) always error, so "control flow cannot depend on tainted
  values"; `taint_atomic` wraps functions. "By construction it is not
  possible for user programs to circumvent taint-wrapping." **~300 LOC;
  ~33% slowdown translated** ("the most straightforward implementation").
  Presented at IBM Zürich (Feb 2007, with Michael Franz): "we have already
  accomplished the state-of-the-art with our prototype." Stated open
  research question: control flow depending on sensitive data "in the
  absence of whole program analysis."
- **Transparent proxies**: objects that masquerade as instances of any
  (builtin) type with a controller function intercepting every operation
  (`make_proxy(controller, type=list)` satisfies `type(l) is list`).
  Motivated by Python's descriptor-based object model making conventional
  proxying "quite challenging" (subclassing builtins doesn't help — internal
  operations reach the opaque inherited part). Implemented for builtins as
  one more multimethod implementation per type; for internal types (frames)
  by type-check-and-fallback. **<300 LOC**, optional at translation time,
  "no significant impact on the rest of the application." Compared to
  .NET's TransparentProxy/RealProxy. Consumers documented: the
  `distributed` prototype (transparent lazy remote objects over an RPC-like
  protocol; only atomic immutables by value; works for builtin instances
  including frames/tracebacks → remote debugging with local-looking remote
  tracebacks; "it is very hard to distinguish a remote object from a local
  one"), and the preferred **orthogonal persistence** design (selective
  transparent interception of changes, judged easier than pruning the
  reachability graph of a suspended computation).
- Summary sentence for both: "in all cases, pervasive changes were not
  required."

### 6.4 Configuration and integration (D13.1)

- The configuration framework distinguishes **option descriptions/schemata**
  (a tree of typed options with defaults, dependency/conflict/suggestion
  relations declared as data, auto-generated command-line parser and
  documentation) from **configurations** (one concrete choice). "Every time
  an option is set, all its dependencies are checked"; values are
  **write-once**. "Checks … are done at translation time: the translated
  PyPy interpreter does not contain any actual checks for configuration
  values and also does not contain the code which would be executed if a
  different value … was used" — options behave like C preprocessor macros.
  Built PyPy-unspecific for reuse.
- **The 1.0 compatibility matrix** (Figure 1 of D13.1) records what actually
  composes:
  - WP06 interpreter optimizations compose with everything — except **tagged
    integers**, which need RTyper support, "cannot work with statically and
    strongly typed high-level target environments such as the CLR and the
    Java VM for fundamental reasons," and at 1.0 worked only with Boehm.
  - WP07: the framework GCs are pointless on JVM/CLR (own GC); the stackless
    transformation "currently only works with the C backend" (a "temporary
    restriction" for OO backends; LLVM excluded "for minor technical
    reasons"); inlining and malloc removal were ported to OO backends; stack
    allocation "cannot work for fundamental reasons."
  - WP08: **the JIT was deliberately supported only in the default
    configuration** (std objspace, Boehm, no stackless, no tagged ints) and
    "can currently only be translated using the C and the LLVM backend,
    since it is built to work with the low level type system only." The
    taint space works with everything *except* the JIT ("some
    StdObjSpace-specific code, which should be fixable easily").
  - WP09: the logic space needs clonable coroutines (stackless + framework
    GC) → C backend only, PyPy GCs only ("unlikely to ever work with an
    object-oriented backend").
  - WP10 (AOP) is purely a parser-level feature, independent of everything —
    with the recorded caveat that "one has to be very careful that aspects
    don't affect code that one does not expect them to affect."
  - High-level backends were "one of the hardest features to integrate"
    (deep RTyper changes — the ootype system — plus new translation steps).
- **Build tooling**: translate.py (step ordering, interactive debugging
  console, flow-graph viewer); an experimental **build farm** (meta server on
  codespeak + donated build servers + request clients, over py-lib execnet;
  results as ZIP links; status web page).
- **Debian packaging**: the source package splits into **pypy-dev, pypy,
  pypy-stackless, pypy-logic, pypy-lib, pypy-doc** (plus maintenance of
  python-codespeak-lib for py.test); flavors exist precisely because
  differently-configured interpreters are different binaries. Debian's
  autobuilders across 11 architectures are cited as free porting/QA.

---
## 7. Research prototypes (D09.1, D10.1, D11.1)

### 7.1 Logic and constraint programming (D09.1)

- **What was built**: logic variables (`W_Var` at interp level; free/bound
  single-assignment; `newvar`/`bind`/`unify`, aliasing between free
  variables) in a dedicated **Logic object space**; dataflow synchronization
  (touching a free variable suspends the coroutine until it is bound;
  `wait`, `wait_needed` for lazy producers; `future` spawning a coroutine
  bound to a result variable); a grammar extension `choice:`/`or:` for
  non-deterministic choice; Oz-style **computation spaces** driven by an
  external solver through an ask/choose/commit protocol; and search by
  **GC-pool coroutine cloning** (§3.8). The constraint engine (CSP as
  (X, D, C); domains, expressions, AllDistinct; solvers written in pure
  Python) has its performance-critical propagation core as a pure RPython
  library, also usable standalone without the logic space.
- **Architecture credits** (the conclusion's own list): the first-class
  object space ("allows to wrap additional semantics around normal Python
  semantics"), RPython ("facilitates writing interpreter-level code"),
  interpreter-level multimethods ("proved to be very convenient" for the
  new builtins), the GC framework's cloning ("impossible with the
  Böhm-Weiser GC"), and coroutines. Verdict quoted verbatim: **"We can
  assert without doubt that this work would have been completely impossible
  within CPython."**
- **Honest status at ship**: "as of this report, the key functionality of
  space cloning still doesn't work"; "operations related to search with
  logic programs don't work … However, the non-concurrent version of the
  constraint solver is usable." The debugging loop is recorded as the
  blocker: a full translation with the logic space takes "almost two hours
  on a fast machine," followed by stepping through "several millions lines"
  of generated C laced with GC root push/pops.
- Recorded design doubts and rejected directions: logic variables' "half
  transparency" wraps nearly every operation in `wait` — "one could
  reasonably argue that the run-time price is too high, and that it is not
  pythonic"; Schulte-style constraint combinators "left out" (involved, and
  measured elsewhere as underperforming); Constraint Handling Rules out of
  scope; cloning-vs-trailing is defended by the Oz group's measurements plus
  the argument that cloning "inherently opens the ability to distribute the
  load amongst several CPUs … whereas trailing does not support parallelism
  at all."
- **OWL/SPARQL application**: an OWL-DL reasoner by conversion to a
  constraint problem (not tableaux); SPARQL front end offering "variables at
  the predicate position, which even Pellet does not"; target ontology LT
  World (~1500 classes, ~15 MB). A full consistency check took **~90
  minutes**; testing was limited to a small manually-corrected subset;
  "various queries produce correct answers, but we are … not satisfied with
  the results yet." The application evaluation is "the only part considered
  non-final," delayed by resolver bugs and the departure of the key
  developer.

### 7.2 AOP, design-by-contract, RPylint (D10.1)

- The enabling mechanisms: the grammar is changeable at runtime and
  `parser.install_compiler_hook(callback)` fires on every AST construction
  (imports and `eval`/`exec`), with read-only visit and mutating
  `node.mutate(ASTMutator)` traversal.
- The pure-Python `aop` module weaves at **import time** by AST mutation:
  Aspect metaclass, before/after/around/introduce advices, AspectC++-style
  point-cut regexes. Documented drawbacks of static weaving: modules with
  existing `.pyc` files bypass the weaver; the importing file itself is
  parsed before the hook installs. Of two ways to inject compiled advice
  code, the indirection through a `__aop__` builtin was chosen over
  reparsing to AST.
- Candid self-judgments: "it is not clear whether working at the AST level
  to weave advices gives significant advantages over the dynamic approach
  used by all other implementations of AOP in Python" — though the AST hook
  is "a very powerful tool … potentially the first step towards adding
  macros to the language." For DbC: "we feel that this approach … is not
  the best one in a language such as Python, and did not push the effort
  too far" — CPython-compatible libraries (PyDBC/pycontract) are called
  "a more pythonic way."
- **RPylint** exists because of the translator's error ergonomics: slow,
  non-incremental, first-error-only, messages "not as explicit as expected,"
  surfacing as stack traces. It is a Pylint extension checking RPython rules
  at source level (banned constructs/builtins/protocols, multiple
  inheritance, type consistency, homogeneous lists, constant globals,
  negative slices), plus a test framework pairing translatable/
  untranslatable snippets with expected messages — proposed as de-facto
  documentation, "one problem with RPython is that the language has no
  specification and is evolving as PyPy is being developed." Its inference
  differs from the toolchain's, so passing RPylint "make[s] it impossible to
  guarantee the success of the translation"; judged useful for beginners and
  porting estimates, not for experts.

### 7.3 Embedded devices (D11.1)

- Five options assessed, with verdicts: (1) full generated interpreter on
  the device — "for now not feasible" (a generated interpreter is "an order
  of magnitude larger" than CPython: ≈5.3 MB vs ≈1.0 MB file size); (2) a
  reduced-language interpreter — "an interesting way to investigate," but
  "nothing in the current code of PyPy allows doing this out-of-the-box"
  (though deriving a reduced object space is argued to be easy in
  principle); (3) an RPython *interpreter* — rejected as a category error
  ("RPython was not designed with an interpreted goal in mind"); (4) using
  language-aspect extensions for safety; (5) **compile RPython applications
  with the toolchain** — "pretty much possible … with very minimal effort";
  this is what the case study did.
- **Case study**: the stdlib HTTP server ported to RPython (rpyhttp) for
  Axis ETRAX 100LX devices (2–8 MB flash, 8–32 MB RAM). Time budget: ~7
  days total — **60% making stdlib code RPython-compliant, 35% improving
  RPython itself** (multi-char strip, tuple comparison, negative indices);
  the server logic "took five minutes." The port catalog is a concrete
  census of RPython friction: rsocket instead of socket; no negative
  indices except -1; one-char single-argument `split`; positional-only `%`
  formatting without `%r`/`%x`; no `*args`/`**kwargs`; no
  `getattr(self, 'do_' + command)` dynamic dispatch (→ if/elif chains);
  file I/O via `os.read`/`os.write` on descriptors; several stdlib
  functions reimplemented (os.path ops, urlparse, quote, escape).
  Result: a binary whose text segment is roughly Boa's size (87 KB vs
  71 KB) with comparable request-serving CPU times.
- **Pain points recorded** (the embedded lens on the whole toolchain):
  exception-check code bloat after every call (C backend returns NULL like
  CPython; C++ exceptions or setjmp/longjmp noted as size levers);
  framework-GC per-call root bookkeeping vs Boehm ("significant overhead
  both in terms of code size and performance"; "the Boehm garbage collector
  is probably the lightest available"); **no modular build** ("cannot link
  … using a minimalistic version of the libc (e.g. uclibc)" without
  libm/libpthread); no real-time or memory-constrained GC; cryptic
  one-at-a-time translation errors; docs not RPython-oriented. "The two
  hardest points are modular compilation and the real-time garbage
  collector."
- Future leads named: **microPyPy** (a reduced interpreter for a subset
  "less limited than RPython" — with the admission that nothing supports
  this yet), toolchain-as-static-checker (stack-depth bounds, recursion
  detection, Stanford-checker-style rules), real-time GC. Overall verdict:
  "a valuable and viable solution for embedded applications on not too
  small devices," but "currently not suitable … for regular Python
  programmers" — RPython "still a bit tedious," libraries missing — "most
  of them are merely a matter of the PyPy project maturing beyond the state
  of a research project."

---
## 8. Process, QA, testing (D01.x, D02.x, D14.2, D14.5)

### 8.1 Testing doctrine (D02.3, D01.1)

- The stated backbone: "Development and research within the PyPy project
  relies on automated testing and a test-driven development approach." D01.1
  names the automated test suite "the main tool for handling exceptions and
  for very quickly identifying and implementing corrective actions" and "the
  primary quality assurance tool and strategy," with the explicit mandate:
  **"It is a requirement to add new tests for each newly added features and
  for each fixed bug."** Policy before commit: run tests and make sure no
  regression occurred; failures are traceable to the source change that
  broke tested behaviour.
- Stated experience claim: "high quality testing tools can positively impact
  development speed and quality." The codebase is described as containing
  "several thousand automated unit and functional tests, running in several
  languages and operating systems."

### 8.2 The test taxonomy (D02.3)

The architecture "requires testing to occur at multiple levels."
Project-specific `py.test` extensions recognize several test types that
"co-exist … and can be uniformly driven by py.test invocations":

1. **Interpreter-level tests** — set up with an Object Space, run directly
   on CPython; test low-level interpreter or translation aspects.
2. **Application-level tests** — run through PyPy's own interpreter on top
   of CPython, and additionally "can run directly on a translated PyPy
   version (which is considerably faster as it avoids the double
   interpretation overhead)."
3. **Compliance tests** — the CPython regression suite, instrumented to run
   either through PyPy's interpreter or on a translated binary. Both the
   getting-started page and the sprint slides state "PyPy passes around 95%
   of CPythons core language regression tests."
4. **JavaScript tests** — ECMA regression tests running on the emerging
   JavaScript interpreter (py.test is "not limited to Python code").
5. **Documentation tests** — website generation plus full link and
   reference-integrity checks, via a py.test extension.

Running everything at once "takes a very long time, and enormous amounts of
memory"; an autotest driver executed tests directory-by-directory and
published nightly summary pages. Automated test runs *and automated
translations* ran each night on HHU-donated machines "to identify
integration problems early" (D02.1); nightly pystone/richards runs guarded
performance regressions (D06.1, §4.1).

### 8.3 py.test and the py lib (D02.3)

- **Rationale vs unittest**: subclass-based unittest extension "often
  intermingles project specific integration code with these advanced
  features and thus prevents re-use from other projects." py.test instead
  lets projects "modify and amend aspects of the testing process without
  requiring intrusive changes to the actual testing code."
- Features as stated: no framework class hierarchy (collection by naming
  convention, conventions modifiable); a **modified `assert` statement**
  reporting the values in a failed assertion ("assertions about truth values
  are regular Python expressions"); automatic stdout capture (debug prints
  can stay in test code permanently); failure introspection down to an
  interactive debugger on the failing frame; all collection/execution/
  reporting behavior customizable per-directory via `conftest.py` (used in
  PyPy to exclude directories and to drive remotely-run Windows GUI tests
  from Linux).
- **Ad-hoc distributed testing** over `py.execnet` (compile-and-execute
  program text on remote hosts over SSH/sockets, objects exchanged via
  Channels, zero remote installation beyond a Python interpreter):
  distributes the suite to multiple hosts — "considerable speed up,"
  especially for tests that translate snippets to C or .NET. A real-time
  progress web application was itself built with PyPy's JavaScript backend.
  Distributed testing "has encouraged developers to perform larger test runs
  before committing." execnet also coordinated translation build jobs and
  SVN replication.
- **apigen** derives argument/return types and call chains by monitoring
  calls during test runs — "hard to obtain statically in a dynamic
  language"; doubles as "an indicator for the quality of the tests."
- The py lib was released separately (**0.9.0, February 2007, MIT license**,
  CPython 2.3–2.5); its collection/state-management concepts were adopted by
  **nose**; described as "one of the major Python testing tools."

### 8.4 Sprint-driven development (D14.2, D01.2-4)

- The dissemination/development strategy is "leveraging the community" via
  **Sprint Driven Development**: one-week face-to-face coding sessions,
  "minimizing many of the risks of traditional distributed and dispersed
  F/OSS development" — agile practices (pair programming, whole team, TDD,
  daily planning), explicitly *not* SCRUM's "sprint."
- Sprints are the main contributor entry point: "a majority of today's PyPy
  contributors entered the project in this way." Newcomer travel was
  reimbursed via the "Physical Partner" model and the "Summer-of-PyPy"
  programme.
- Each newcomer sprint opens with the tutorial "PyPy — Crash Course/Sprint
  Intro" (**42 pages, ~1 hour**; presented at **9 public sprints to ~50
  newcomers** during the project). Its stated philosophy: teach "just enough
  to support the main work in the sprint which is coding — 'learning by
  doing'"; the overview must be "almost immediately followed up by pair
  programming with someone more experienced."
- Newcomer testimonials (four Summer-of-PyPy participants) converge on: the
  online docs give a broad view but are insufficient alone; **the IRC
  channel (#pypy) and sprint pairing made the decisive difference** ("what
  definitely made the difference was the IRC channel" — Cuni; "Otherwise I
  don't think I would be able to enter the PyPy project at all" —
  Fijalkowski).
- **Scale**: 7 sprints before EU funding (2003–2004); **18 sprints during**
  (Leysin 2005-01 through Hildesheim 2007-03), participant counts 6–24;
  16 sprint reports doubled as the external newsletter.

### 8.5 QA plan and coordination structure (D01.1, D01.2-4)

- D01.1 states the tension openly: "The agile practices being implemented
  within the project collide with the traditional project management
  approach of detailed segmented plans." The response is a "minimalistic
  approach" — "document procedures and features only when deemed necessary,"
  with ~6-monthly **Internal Review Workshops ("Learning Loops")** revising
  the procedures themselves.
- QA is grouped into four categories: automated procedures (VCS, test
  suites, tracking infra), open procedures (peer review with few or no
  access restrictions), role procedures, and agile procedures
  (sprint-driven development plus layered evaluation loops). Risk strategy:
  "cyclic, iterative approaches … allow for early risk identification" —
  sprints (project level), monthly consortium meetings, weekly 30-minute
  **pypy-sync** IRC meetings + Technical Board (technical level).
- Coordination record: **27 pypy-sync meetings, 15 Technical Board
  meetings, 17 Consortium meetings (3 physical), 7 Management Team
  meetings** during funding. Internal reporting ran primarily on
  **SVN-commit-notification emails** (every change traceable to author +
  log) plus IRC; monthly timesheets in a shared repository; "Deviations
  larger than 10% need to be explained on consortium level." Conflict
  resolution: consensus first, then vote, then arbitration. Technical
  development requests/bugs in a public Roundup issue tracker.
- Notable structural fact: none of the Technical Board's core developers
  represented partners holding consortium-management budget; EU deliverable
  reports were run by a rotating "report release manager" — "a technique
  borrowed from open source software release management."

### 8.6 Infrastructure and openness (D02.1)

- Everything on self-run **codespeak.net** (chosen over SourceForge for
  "more control and transparency"; Subversion was not readily available
  elsewhere early on). Stated open-access policy: anyone can work in private
  areas then contribute; "hardly any contributor can do permanent damage to
  a fully versioned file system"; "There were no incidents of abuse of this
  open policy"; infrastructure "should be very careful to not impose
  restrictions which lead to unnecessary overheads." Over 200 registered
  users by March 2007; very few outages in 28 months, "never resulting in
  data damage or loss."
- Repository resilience: an svn-sync-repo mirror replaying revisions at
  ~1-minute delay, daily tiered backups, single-login hooks. Website pages
  generated from reStructuredText under version control; video
  documentation distributed by BitTorrent (**6,955 successful downloads of
  40 torrents by 2007-02**; "well over 7,500" by March).
- **Growth over the 28 months**: code **~30,000 → ~340,000 lines**, test
  code **~8,000 → ~82,000 lines**; pypy-dev subscribers ~150 → ~330 (324 on
  2007-02-19); ~40,000 monthly site visits; 250,000 IRC lines with ~20
  people present in #pypy on average.

### 8.7 Release scheme (D02.2)

- Two-track model: continuous public SVN ("the 'revisions' were usually
  consistent in the sense that all automated tests passed. We developed
  tools to track this consistency and determine which revision has
  regressions") plus **six formal releases** "to settle and formally
  announce … making sure that we have appropriate documentation and a clear
  message":

| Release | Date | Stated content |
|---|---|---|
| 0.6 | 2005-05-20 | core Python interpreter (on CPython) |
| 0.7 | 2005-08-28 | first *translated* Python interpreter |
| 0.8 | 2005-11-03 | more translation aspects, optimizations |
| 0.9 | 2006-06-26 | stackless, extension compiler, framework GCs, logic programming |
| 0.99 | 2007-02-17 | high-level backends, optimizations |
| 1.0 | 2007-03-27 | optimizations, JIT compiler generator, transparent proxies, AOP |

- A single mainline **dist** branch with derived release tags "proved
  sufficient"; experimental work usually merged back into mainline. The
  release process became "increasingly refined and automated," to the point
  that "the effort to perform such a release lies mostly in considering and
  expanding our test sets, and updating and reviewing documentation."
  Release managers temporarily control the development branch before a
  release.
- Recorded scaling concern (March 2007): "as the number of backends and PyPy
  features and options increases, it is not easily possible to ensure that
  dist is stable with respect to an increasing set of test combinations" —
  the planned answer was a trunk/dist split (develop on trunk, copy stable
  tested snapshots to dist), with mainline integrity resting on "a group of
  trusted contributors."

### 8.8 Documentation structure (D14.2)

Three deliberate documentation strands: **user documentation**
(getting-started doubling as release notes and online tutorial — a quoted
user calls it "about the clearest, best such page I have ever seen"),
**project documentation** (talks, EU reports published for community
feedback, sprint reports, video archive — a "whole brained" approach
documenting process as well as code, also used by academic researchers), and
**source-code documentation** (per-component: why the component is needed,
what services it performs, main design decisions, with examples and tests —
positioned as "second-line support" behind getting-started). The
getting-started source-tree map (interpreter/, objspace/std/, translator/,
annotation/, rpython/ with per-type `rxxxx.py` files) is the canonical
"where to start reading" index.

---
## 9. Deviations, difficulties and retrospective judgments (D14.x, Final)

### 9.1 Deviations the project itself recorded

1. **Duration extended 24 → 28 months** (requested during phase 2). Stated
   reasons: resource-deployment limitations, following up Review
   recommendations, and the changed (more specialized, commercially driven)
   nature of sprints and workshops. "Fortunately, the Commission reacted with
   flexibility when we requested changes to the work plan, project duration
   and also prepayment calculations."
2. **Key-person health incident**: Tismerysoft's key person was stopped by
   serious health issues from early June 2006, halting that partner's WP07
   (massive parallelism) work; other partners "collaboratively completed the
   most crucial WP7 results" for the 0.9 release; the WP07 report shipped
   2–3 months late; the project prepared "for the situation where this
   partner cannot return to the project at all."
3. **Deliverable restructuring**: Amendment 4 restructured 58 deliverables
   into 21 contractual deliverables; five amendments total. Interim reports
   (D07.1, D09.1) were invented mid-project, "endorsed by reviewers, to
   provide status before the actual deliverable is due," and were "also used
   by community members to prepare for sprints."
4. **Phase-3 work pulled forward** (extension compiler WP03, validation WP12,
   integration/configuration WP13) on commercial/community demand and for
   post-funding sustainability.
5. **Extension-module support consciously deprioritized** in favour of the
   research vision; PyPy 1.0 consequently "still not generally usable" (§1.5).
6. RPython deliberately **not generalized** for standalone/commercial use
   during the project ("it was not feasible to focus on improving and
   enriching RPython for other purposes"), despite commercial interest in
   RPython as "a 'better Java'".
7. **Unforeseen directions added by the community**: JavaScript backend and
   the beginnings of a JavaScript interpreter, AJAX tooling, CLI/.NET and JVM
   backends, a Scheme interpreter — "resulting in new features and
   directions, not foreseen at the start of the EU project." The reports
   treat this as positive validation, "welcomed and supported by the PyPy
   core group, because it served both as early validation of PyPy's
   architecture and broadening its overall applicability."
8. **EU-style per-partner effort tracking deviated** because "the bulk of the
   development work done in PyPy was self-organized and peer-driven" — "the
   tracking of planned work and cost consumption per partner did deviate both
   when it came to development work and management work which had to be
   explained and justified."

### 9.2 Funding-mechanics difficulties (Final Report §1.6)

- Paying individual OSS contributors under FP6 was "very hard": the
  "Physical Partner" model required contract amendments and full consortium
  duties — "far too heavy an obligation for experts that only contribute one
  or a few intense work weeks and only receive travel reimbursement." The
  lightweight **"Summer of PyPy"** model (inspired by Google Summer of Code;
  five participants approved) was invented as the fix and "was very
  successful," with the side effect of reaching undergraduate/postgraduate
  students. Their work triggered the JavaScript interpreter, the
  JavaScript/AJAX mapping, the CLI backend, and the JVM backend.
- The 50% co-financing rule was "burdensome" for the SME-majority consortium:
  "It can be hard for SME's to focus on the long-term goals of serious
  research and development work, while simultaneously having to make money to
  finance its own share of the costs." The report recommends a higher SME
  funding rate for FP7.
- Transnational hiring was effectively blocked ("huge differences in
  taxation, labour laws and general lack of support"); consultant hiring
  counts as sub-contracting and "is hard to use ... in practice." "We have on
  several occasions had to forgo hiring people that would have improved
  project performance."

### 9.3 Retrospective judgments on the process

- The central tension: "the project experienced a growing need to balance
  contractual obligations with a community wanting to follow its own ideas
  and priorities. It would become more difficult to combine the mentoring
  approach with the complex and time-critical contractual tasks within one
  sprint." Reaction: sprints became **pre-targeted** either at internal
  milestones or at newcomer mentoring.
- "The project succeeded because of the 'people factor' and because of the
  huge amount of work and energy core people was prepared to invest."
- Sprint-driven development is judged by its own report (D14.5) to be "not a
  methodology, but rather a practice, and in general a practice is not
  inherently agile" — but PyPy's implementation of it "was indeed an agile
  implementation" and "the primary success factor for managing to balance
  community interaction and demands with a plan-driven frame work."
- Post-funding forecast (written before the end): sprint frequency will drop
  to "maybe not more than four sprints per year"; development "will likely be
  structured in a more typical Open Source style: advances will be driven by
  somewhat separate interests, namely engineering, academic and commercial
  interests. A group of trusted individual contributors has been installed to
  keep conceptual integrity."

External recognition quoted in D14.4 (a Ruby developer): "doing the types of
things the PyPy team has done is HARD. So hard that even with great funding,
lots of experience, and quite a bit of time, they still aren't where they
might want to be yet. And therefore, for some of these other teams, it might
make a lot more sense to harness the work the PyPy team has already done
instead of doing it all on their own."

---
## 10. Cross-cutting statements of philosophy (verbatim anchors)

- "The primary goal is to allow us to implement the full Python language only
  once, as an interpreter, and derive interesting tools from it." (D05.1)
- "The source of our Standard Interpreter is an executable specification of
  the Python language." (D07.1)
- "Static analysis is and remains slightly fragile … This is also a reason why
  we believe that dynamic analysis is ultimately more powerful." (D05.1 §8.1)
- "Our approach gives us flexibility and lets us choose various aspects at
  translation time instead of encoding them into the implementation itself."
  (D05.3)
- "It is surely better to not write the reference count manipulations at all.
  … once it is done and tested, it just works without further effort." (D05.4)
- "A JIT compiler generator renders the issue of performance mostly orthogonal
  to the evolution and maintenance of the language interpreter." (D14.4)
- "Ideally, turning an interpreter into a JIT compiler is only a matter of
  adding a few hints. In practice, the current JIT generation framework has
  many limitations and rough edges requiring workarounds." (D08.1 App. 1)
- "Interpreters for dynamic languages can be implemented on a high level of
  abstraction, yet be brought to run at least as efficiently as today's common
  lower level implementations." (D14.4)
- "It is a requirement to add new tests for each newly added features and for
  each fixed bug." (D01.1)
- "The flexibility of PyPy's Python implementation has been essential to the
  task of experimenting with and especially assessing these new
  implementations, vindicating the effort we put in to allow this
  flexibility." (D06.1)

---

## Appendix A. Deliverable index

| ID | Title (abbrev.) | Content |
|---|---|---|
| D01.1, D01.2-4 | QA plan; project organization | Minimal-procedure QA; TDD as primary QA tool; roles, meetings, tracking |
| D02.1, D02.2, D02.3 | Tools/website; release scheme; testing framework | codespeak infra; six releases + continuous SVN; py.test, test taxonomy, distributed testing |
| D03.1 | Extension compiler | 2.4.1 target; mixed modules; CPy object space; rctypes and its pivot |
| D04.1–D04.4 | Standard interpreter | Object spaces; app/interp levels; W_Objects, multimethods; parser/compiler rewrites; thunk space |
| D05.1–D05.4 | Translation | Flow space, annotator (formal model), RTyper, backends; translation aspects; measured aspect costs |
| D06.1 | Core optimizations | Benchmark methodology; multidicts, method optimizations, dispatch, per-optimization numbers |
| D07.1 | Parallelism & translation aspects | Stackless transform, coroutines/tasklets/greenlets, pickling/cloning, composability; GC framework, GC transformer, all GC numbers |
| D08.1, D08.2 | JIT | JIT generator: BTA, timeshifting, promotion, virtuals, virtualizables; 1.47× gcc result; open issues |
| D09.1 | Logic/constraints | Logic object space, computation spaces, GC-cloning search; OWL/SPARQL; cloning not working at ship |
| D10.1 | AOP | AST hooks, import-time weaving, DbC; RPylint |
| D11.1 | Embedded | Five options; RPython web-server case study; size/modularity blockers |
| D12.1 | High-level backends & prototypes | ootype; GenCLI/GenJVM/GenJS; taint space; transparent proxies |
| D13.1 | Integration & configuration | Translation-time config system; 1.0 compatibility matrix; build farm; Debian |
| D14.1–D14.5, Final | Milestones, process, final report | Phase results, deviations, sprint methodology, l×o×p, final self-assessment |

### A.1 Source files

Exact filenames in the source directory (`eu-report/`), one per deliverable:

| ID | File |
|---|---|
| D01.1 | `D01.1-Create_QA_plan_for_the_project.pdf` |
| D01.2-4 | `D01.2-4_Project_Organization-2007-03-28.pdf` |
| D02.1 | `D02.1_Development_Tools_and_Website-2007-03-21.pdf` |
| D02.2 | `D02.2_Release_Scheme-2007-03-30.pdf` |
| D02.3 | `D02.3_Testing_Framework-2007-03-23.pdf` |
| D03.1 | `D03.1_Extension_Compiler-2007-03-21.pdf` |
| D04.1 | `D04.1_Partial_Python_Implementation_on_top_of_CPython.pdf` |
| D04.2 | `D04.2_Complete_Python_Implementation_on_top_of_CPython.pdf` |
| D04.3 | `D04.3_Report_about_the_parser_and_bytecode_compiler.pdf` |
| D04.4 | `D04.4_Release_PyPy_as_a_research_tool.pdf` |
| D05.1 | `D05.1_Publish_on_translating_a_very-high-level_description.pdf` |
| D05.2 | `D05.2_A_compiled,_self-contained_version_of_PyPy.pdf` |
| D05.3 | `D05.3_Publish_on_implementation_with_translation_aspects.pdf` |
| D05.4 | `D05.4_Publish_on_encapsulating_low_level_language_aspects.pdf` |
| D06.1 | `D06.1_Core_Optimizations-2007-04-30.pdf` |
| D07.1 | `D07.1_Massive_Parallelism_and_Translation_Aspects-2007-02-28.pdf` |
| D08.1 | `D08.1_JIT_Compiler_Release-2007-04-30.pdf` |
| D08.2 | `D08.2_JIT_Compiler_Architecture-2007-05-01.pdf` |
| D09.1 (interim) | `D09.1_Constraint_Solving_and_Semantic_Web-interim-2007-02-28.pdf` |
| D09.1 (final) | `D09.1_Constraint_Solving_and_Semantic_Web-2007-05-11.pdf` |
| D10.1 | `D10.1_Aspect_Oriented_Programming_in_PyPy-2007-03-22.pdf` |
| D11.1 | `D11.1_PyPy_for_Embedded_Devices-2007-03-26.pdf` |
| D12.1 | `D12.1_H-L-Backends_and_Feature_Prototypes-2007-03-22.pdf` |
| D13.1 | `D13.1_Integration_and_Configuration-2007-03-30.pdf` |
| D14.1 | `D14.1_Report_about_Milestone_Phase_1.pdf` |
| D14.2 | `D14.2_Tutorials_and_Guide_Through_the_PyPy_Source_Code-2007-03-22.pdf` |
| D14.3 | `D14.3_Report_about_Milestone_Phase_2-final-2006-08-03.pdf` |
| D14.4 | `D14.4_Report_About_Milestone_Phase_3-2007-05-01.pdf` |
| D14.5 | `D14.5_Documentation_of_the_development_process-2007-03-30.pdf` |
| Final | `PYPY-EU-Final-Activity-Report.pdf` |

## Appendix B. Key quantitative anchors

Translation and aspects:

- Annotating PyPy: ~20k blocks / ~4k functions, ~5 min, rules re-applied
  20–40×.
- Aspects at 0.8: stackless +8%/+28%; refcount 2× vs Boehm; thunk +6%/+13%;
  composition prediction 1.14× vs measured 1.15×.
- Stackless at 0.9-era: +17–28% time, 2.1–2.4× code; struct-locals
  alternative +60–64%.
- GCs vs CPython: Boehm 4.4–5.8×; mark-and-sweep 5.7–5.8×; refcount 7.7–7.8×.
- Inlining +52–64%; +malloc removal → 1.03–1.13 of default; escape analysis
  redundant with malloc removal; tagged pointers +7% (with const-folding);
  malloc removal kills ~20% of mallocs.
- First full translation: 22× (richards) to ~460× (pystone) slower than
  CPython; pypy-c-0.7 with refcounting 104.7× on pystone.

Interpreter optimizations:

- All interpreter optimizations: >2× (richards 2.5×).
- Multidicts +15–40%; three method optimizations combined: richards 0.53
  (47% faster); precomputed `__xyz__` lookups: disabling costs 22–32%;
  CALL_LIKELY_BUILTIN 11–19%.
- Multiple Row Displacement: −2.5 MB, almost half the binary, a few percent
  slower (but 15% slower than double dispatch on CLI/Mono).
- SharedDict: 1M two-attribute instances 140 MB vs 266 MB baseline (CPython
  202 MB, `__slots__` 96 MB).

JIT and backends:

- JIT on arithmetic: 1.47× unoptimized gcc (target was 1.5×); Psyco 1.00×.
- pypy-cli ≈6.8× IronPython = 1.8 (backend) × 2 (interpretation) × 1.8
  (objspace).
- 1.0 vs CPython (no JIT): richards 1.17×, pystone 1.55×, memory-heavy 5–8×.
- Binary size: pypy-c ≈5.3 MB vs CPython ≈1.0 MB.
- Taint space: ~300 LOC, ~33% slowdown; transparent proxies <300 LOC.

Process:

- Code ~30k → ~340k lines; test code ~8k → ~82k lines over 28 months.
- ~95% of CPython core regression tests pass; PyPy-on-CPython ~2000× slower
  than CPython.
- 18 sprints during funding (6–24 participants); 6 releases; 27 pypy-sync
  meetings.
