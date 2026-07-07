#!/usr/bin/env python3
"""Forwarding shim: extraction drivers now live per-repo.

The pyre crate table moved to `pyre/scripts/extract-llbc.py` (external consumer
repos carry their own drivers). This shim preserves the historical
`scripts/extract-llbc.py` entry point that CI, `build.rs`, and `check.py`
invoke, forwarding argv verbatim.
"""

import os
import sys
from pathlib import Path

driver = Path(__file__).resolve().parents[1] / "pyre" / "scripts" / "extract-llbc.py"
os.execv(sys.executable, [sys.executable, str(driver), *sys.argv[1:]])
