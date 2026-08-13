# Ubuntu 24.04 arm64 repro image

This image fixes the Linux userspace used to reproduce Linux-only failures on
an Apple silicon host without Rosetta, so the build runs at native speed. Use
`tools/ubuntu24-amd64-repro` instead when the failure is arch-specific or when
confirming against the amd64 CI runner.

Like the amd64 image, it does not copy the repository in; mount the worktree at
runtime so local edits and build artifacts are visible.

Build with Apple `container`:

```bash
container build --platform linux/arm64 -m 8G -c 4 --progress plain \
  -t pyre-ubuntu24-arm64-repro \
  tools/ubuntu24-arm64-repro
```

Run from the repository root:

```bash
mkdir -p "$(dirname "$PWD")/.pyre-build"
container run --rm --platform linux/arm64 -m 20G -c 4 \
  --mount type=bind,source="$(pwd)",target=/workspace/pyre \
  --mount type=bind,source="$(dirname "$PWD")/.pyre-build",target=/workspace/.pyre-build \
  pyre-ubuntu24-arm64-repro
```

The second mount keeps the shared Charon cache outside the worktree and reuses
it across sibling worktrees and container runs.

Inside the container:

```bash
scripts/install-charon.sh
python3 scripts/extract-llbc.py
cargo build --release -p pyrex --bin pyre-dynasm \
  --no-default-features --features dynasm
python3 pyre/cpython_tests/run.py --binary ./target/release/pyre-dynasm
```

The extraction refuses to stamp its artefacts when the tree moves mid-build, so
keep other sessions off the worktree while it runs.
