#!/usr/bin/env python3
"""Charon fixture inspector — dumps shape of an .llbc / .ullbc JSON file.

Run on either form (`charon cargo` -> .llbc, `charon cargo --ullbc` -> .ullbc).
Reads JSON, prints crate summary, lists local fun_decls, and for selected
functions walks the basic-block graph (ULLBC only) or structured body (LLBC).

Usage:
    python3 inspect_llbc.py <file.ullbc> [fun_name_filter ...]

Examples:
    python3 inspect_llbc.py corpus.ullbc
    python3 inspect_llbc.py corpus.ullbc branch_loop_sum desugar_mix
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def name_path(item) -> str:
    if item is None:
        return ""
    parts = []
    for p in item.get("item_meta", {}).get("name", []):
        if isinstance(p, dict):
            if "Ident" in p:
                parts.append(p["Ident"][0])
            else:
                parts.append("<" + next(iter(p)) + ">")
    return "::".join(parts)


def term_summary(term) -> str:
    k = term["kind"]
    if isinstance(k, str):
        return k
    op = next(iter(k))
    payload = k[op]
    if op == "Goto":
        return f"Goto->bb{payload.get('target') if isinstance(payload, dict) else payload}"
    if op == "Call":
        return (
            f"Call(target=bb{payload.get('target')}, "
            f"unwind=bb{payload.get('on_unwind')})"
        )
    if op == "Assert":
        return (
            f"Assert(cont=bb{payload.get('target')}, "
            f"unwind=bb{payload.get('on_unwind')})"
        )
    if op == "Drop":
        return (
            f"Drop(cont=bb{payload.get('target')}, "
            f"unwind=bb{payload.get('on_unwind')})"
        )
    if op == "Switch":
        targets = payload.get("targets")
        if isinstance(targets, dict):
            tk = next(iter(targets))
            tv = targets[tk]
            if tk == "If" and isinstance(tv, list) and len(tv) >= 2:
                return f"Switch.If(then=bb{tv[0]}, else=bb{tv[1]})"
            if tk == "SwitchInt" and isinstance(tv, list) and len(tv) >= 3:
                cases = ",".join(f"bb{c[1]}" for c in tv[1])
                return f"Switch.Int(cases=[{cases}], default=bb{tv[2]})"
            return f"Switch[{tk}]"
        return "Switch[?]"
    return op


def summarize(path: str, name_filters: list[str]) -> None:
    with open(path) as f:
        doc = json.load(f)
    t = doc["translated"]
    print(f"file:               {path}")
    print(f"charon_version:     {doc.get('charon_version')}")
    print(f"crate_name:         {t['crate_name']}")
    print(f"has_errors:         {doc.get('has_errors')}")

    n_fun = len(t["fun_decls"])
    n_fun_bodies = sum(1 for f in t["fun_decls"] if f is not None)
    n_fun_err = sum(
        1
        for f in t["fun_decls"]
        if f and isinstance(f.get("body"), dict) and "Error" in f["body"]
    )
    print(
        f"fun_decls:          {n_fun} total ({n_fun_bodies} with body slot, "
        f"{n_fun - n_fun_bodies} opaque refs, {n_fun_err} translation-error bodies)"
    )
    print(f"type_decls:         {len(t['type_decls'])}")
    print(f"trait_decls:        {len(t['trait_decls'])}")
    print(f"trait_impls:        {len(t['trait_impls'])}")
    print(f"global_decls:       {len(t['global_decls'])}")
    print(f"files:              {len(t['files'])}")

    # Crate-root bucket
    buckets: Counter[str] = Counter()
    for f in t["fun_decls"]:
        if f is None:
            continue
        np = name_path(f)
        root = np.split("::", 1)[0] if np else "?"
        buckets[root] += 1
    print("\nfun_decls by crate root (top 10):")
    for k, c in buckets.most_common(10):
        print(f"  {k:30s} {c}")

    # Per-function detail
    print("\n--- per-function bb counts (local crate, top 20 by BB count) ---")
    local_root = t["crate_name"]
    local = []
    for f in t["fun_decls"]:
        if f is None:
            continue
        np = name_path(f)
        if not np.startswith(local_root):
            continue
        body = f.get("body")
        if isinstance(body, dict):
            if "Unstructured" in body:
                local.append((len(body["Unstructured"]["body"]), np))
            elif "Structured" in body:
                local.append((1, np))  # structured = 1 nested block
    local.sort(reverse=True)
    for n, np in local[:20]:
        print(f"  {n:5d}  {np}")

    # Filtered detailed dump
    if name_filters:
        print("\n--- detailed BB dumps for filter matches ---")
        for f in t["fun_decls"]:
            if f is None:
                continue
            np = name_path(f)
            if not any(filt in np for filt in name_filters):
                continue
            body = f.get("body")
            if not isinstance(body, dict) or "Unstructured" not in body:
                print(f"\n## {np} (no Unstructured body)")
                continue
            u = body["Unstructured"]
            print(f"\n## {np}")
            print(f"   arg_count={u['locals']['arg_count']}, "
                  f"locals={len(u['locals']['locals'])}, "
                  f"BBs={len(u['body'])}")
            for i, bb in enumerate(u["body"]):
                print(
                    f"   bb{i:2d}: {len(bb['statements'])} stmts  →  "
                    f"{term_summary(bb['terminator'])}"
                )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    summarize(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
