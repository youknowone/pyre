<!--
Thank you for contributing to Pyre.

Pyre is still a project that relies heavily on AI-assisted development.

For effective collaboration, please follow these rules.

1. The submitter is ultimately responsible for both the code and the discussion. Do not submit generated code or generated discussion without review.
2. Clearly separate the author's own judgment from generated suggestions. Do not submit generated discussion by itself without the author's own assessment.
-->

## Summary


## Self-review

<!--
Did you use AI to create this patch?
If the patch is not AI-generated, write: "This patch is not AI-generated."
Otherwise, please fill out this section.

Note: Do not run the review in the same session that generated the code.

If you are using Claude, a quick local review is `/parity to upstream/main`.

The currently recommended prompt for gpt-6.1-sol will be run automatically when this is ready for review.
The prompt is:

----
Assess by static analysis whether our changes in git diff main are equivalent
to the corresponding RPython/PyPy source code. The RPython and PyPy sources
are available locally.

If anything was ported incorrectly, report every instance in detail. After
collecting all differences, organize the report into separate sections:

1. Cases where our patch regressed PyPy parity compared to main
2. Other mismatches introduced by our patch
3. Mismatches that already existed before this patch
4. Structural adaptations

Exceptions: some differences cannot be ported 1:1 because of Python 3.11 vs
3.14 differences, opcode mismatches caused by using a CPython-compatible
compiler, GIL/free-threading differences, and fundamental implementation-
language differences between RPython and Rust. Mark those separately under
“Structural adaptations.”
-->

- [ ] I fully resolved all reasonable code review comments from Codex and CodeRabbit.
  - [ ] Auto-review section 1 is clear. This check is mandatory.
  - [ ] Auto-review section 2 is clear. If this is not checked, please add a comment explaining why.
- One of checkbox below must be checked.
  - [ ] I added `Assisted-by` to commit messages to the commits AI wrote.
  - [ ] I did not use AI to write the code of this patch.

