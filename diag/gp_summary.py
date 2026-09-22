"""Summarize a wasmtime GuestProfiler (Firefox processed format) JSON: top self and inclusive functions."""

import collections
import json
import sys


def main(path, top=30):
    d = json.load(open(path))
    t = d["threads"][0]
    strings = t["stringArray"]
    frames, funcs, stacks, samples = t["frameTable"], t["funcTable"], t["stackTable"], t["samples"]
    weights = samples.get("weight") or [1] * samples["length"]
    weights = [w if w is not None else 1 for w in weights]
    deltas = samples.get("threadCPUDelta") or [None] * samples["length"]
    if not any(deltas):
        deltas = [None] * samples["length"]

    def fname(frame):
        return strings[funcs["name"][frames["func"][frame]]]

    self_w = collections.Counter()
    incl_w = collections.Counter()
    total = 0
    for i in range(samples["length"]):
        s = samples["stack"][i]
        w = weights[i]
        if deltas[i] is not None:
            w = deltas[i]
        total += w
        if s is None:
            self_w["<no stack>"] += w
            continue
        self_w[fname(stacks["frame"][s])] += w
        seen = set()
        while s is not None:
            n = fname(stacks["frame"][s])
            if n not in seen:
                incl_w[n] += w
                seen.add(n)
            s = stacks["prefix"][s]
    print(f"samples={samples['length']} total_weight={total}")
    print("-- self --")
    for n, w in self_w.most_common(top):
        print(f"{100 * w / total:6.2f}% {n[:140]}")
    print("-- inclusive --")
    for n, w in incl_w.most_common(top):
        print(f"{100 * w / total:6.2f}% {n[:140]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
