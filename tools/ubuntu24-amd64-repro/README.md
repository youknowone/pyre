# Ubuntu 24.04 amd64 repro image

This image fixes the Linux userspace used to reproduce Ubuntu-only
`pyre-cranelift` failures. It intentionally does not copy the repository into
the image; mount the worktree at runtime so local edits and build artifacts are
visible.

Build with Apple `container`:

```bash
container build --platform linux/amd64 -m 8G -c 4 --progress plain \
  -t pyre-ubuntu24-amd64-repro \
  tools/ubuntu24-amd64-repro
```

Run from the repository root:

```bash
mkdir -p "$(dirname "$PWD")/.pyre-build"
container run --rm --platform linux/amd64 --rosetta -m 20G -c 4 \
  --mount type=bind,source="$(pwd)",target=/workspace/pyre \
  --mount type=bind,source="$(dirname "$PWD")/.pyre-build",target=/workspace/.pyre-build \
  pyre-ubuntu24-amd64-repro
```

The second mount keeps the shared Charon cache outside the worktree and reuses
it across sibling worktrees and container runs.

Inside the container:

```bash
scripts/install-charon.sh
scripts/extract-llbc.sh pyre-object pyre-interpreter
cargo build --release -p pyrex --bin pyre-cranelift \
  --no-default-features --features cranelift
RUST_BACKTRACE=1 ./target/release/pyre-cranelift pyre/bench/synth/bool_compare.py
```
