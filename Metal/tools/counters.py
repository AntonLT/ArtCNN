#!/usr/bin/env python3
"""Summarize GPU limiter counters from an xctrace recording made with a Performance Limiters template.

counters.py <trace> : mean of each limiter/utilization counter over samples with compute work in flight.
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def table(trace, schema):
    xml = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
                          f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'],
                         capture_output=True, check=True).stdout
    root = ET.fromstring(xml)
    cols = [c.findtext("mnemonic") for c in root.iter("col")]
    ids = {}
    for row in root.iter("row"):
        values = []
        for el in row:
            if "ref" in el.attrib:
                values.append(ids[el.attrib["ref"]])
                continue
            v = el.attrib.get("fmt", el.text) if el.tag in ("gpu-counter-name", "string") else el.text
            if "id" in el.attrib:
                ids[el.attrib["id"]] = v
            values.append(v)
        yield dict(zip(cols, values))


trace = sys.argv[1]
names = {r["counter-id"]: r["name"] for r in table(trace, "gpu-counter-info")}
samples = defaultdict(dict)
for r in table(trace, "gpu-counter-value"):
    samples[(r["ring-buffer-index"], r["sample-index"])][names[r["counter-id"]]] = float(r["value"])

busy = [s for s in samples.values() if s.get("Compute SIMD Groups Inflight", 0) > 0]
print(f"{len(busy)} busy samples of {len(samples)}")
keys = ("Limiter", "Utilization", "Occupancy", "Bandwidth", "Inflight", "Miss Rate")
for n in sorted({n for s in busy for n in s}):
    if any(k in n for k in keys):
        vals = [s[n] for s in busy if n in s]
        print(f"{n:45s} {sum(vals) / len(vals):10.2f}")
