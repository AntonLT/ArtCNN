#!/usr/bin/env python3
"""Measure GPU and package power of the Metal and libplacebo/MoltenVK implementations.

energy.py <input.f32> <width> <height> [seconds] [fps,fps,...]

Frame rates default to "120,unpaced"; "unpaced" renders frames back to back.

Uses `macmon pipe` (IOReport, no sudo) sampled every 200 ms. For each workload the power is
averaged over its steady part (skipping the first second and the last half second) and
reported above an idle baseline. Workloads alternate between implementations and repeat.
"""

import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "..", "build")
METAL = [os.path.join(BUILD, "artcnn_metal_cli"), os.path.join(HERE, "..", "ArtCNN_C4F16_DS.metal")]
PLACEBO = [os.path.join(BUILD, "plbench"), os.path.join(HERE, "..", "..", "GLSL", "ArtCNN_C4F16_DS.glsl")]
FIELDS = ("gpu_power", "cpu_power", "all_power", "sys_power")

samples = []


def sampler(proc):
    for line in proc.stdout:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        samples.append((time.monotonic(), {k: d[k] for k in FIELDS}))


def mean_between(t0, t1):
    window = [s for t, s in samples if t0 <= t <= t1]
    return {k: sum(s[k] for s in window) / len(window) for k in FIELDS}, len(window)


def run(name, cmd, env):
    t0 = time.monotonic()
    out = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True).stdout.strip()
    t1 = time.monotonic()
    power, n = mean_between(t0 + 1.0, t1 - 0.5)
    return name, out, power, n


def main(input_path, w, h, seconds, rates):
    mon = subprocess.Popen(["macmon", "pipe", "-i", "200"], stdout=subprocess.PIPE, text=True)
    threading.Thread(target=sampler, args=(mon,), daemon=True).start()
    time.sleep(1.5)
    t0 = time.monotonic()
    time.sleep(8)
    idle, _ = mean_between(t0, time.monotonic())
    print("idle: " + ", ".join(f"{k} {v:.2f} W" for k, v in idle.items()))

    args = [input_path, str(w), str(h), "/tmp/energy_out.f32"]
    workloads = []
    for rep in range(2):
        for rate in rates:
            fps, label = ("100000", "unpaced") if rate == "unpaced" else (rate, f"paced {rate} fps")
            for name, tool in (("metal", METAL), ("moltenvk", PLACEBO)):
                workloads.append((f"{name} {label} #{rep + 1}", tool + args, {"ARTCNN_FPS": fps, "ARTCNN_SECONDS": str(seconds)}))
    for label, cmd, env in workloads:
        time.sleep(3)  # let clocks and power settle between runs
        name, out, power, n = run(label, cmd, env)
        frames = int(out.split(":")[1].split()[0])
        extra = {k: power[k] - idle[k] for k in FIELDS}
        per_frame_mj = extra["gpu_power"] * seconds * 1000.0 / frames
        print(f"{name:28s} {out.split(': ', 1)[1]:32s} GPU {power['gpu_power']:.2f} W "
              f"(+{extra['gpu_power']:.2f} over idle), CPU {power['cpu_power']:.2f} W, "
              f"package {power['all_power']:.2f} W, system {power['sys_power']:.2f} W, "
              f"GPU energy above idle {per_frame_mj:.2f} mJ/frame, {n} samples")
    mon.terminate()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
         float(sys.argv[4]) if len(sys.argv) > 4 else 15.0,
         sys.argv[5].split(",") if len(sys.argv) > 5 else ["120", "unpaced"])
