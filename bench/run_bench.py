#!/usr/bin/env python3
"""Run the NeuroZip benchmark and write `results/results.json`.

Every compressor, NeuroZip included, is run the same way: as a command-line
process reading a file and writing a file. For each run we record

* output size in bytes (the whole file the tool wrote, all metadata included),
* wall-clock time,
* CPU time (user + system) of the process,
* peak resident memory of the process,

all taken from the operating system (`wait4` resource usage). Each timed run
is repeated and the median wall time is reported. Every output is
decompressed again and compared with the original by SHA-256.

Timed runs execute one at a time. Ablation runs measure sizes only, so they
run in parallel.

Standard library only.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPORA = HERE / "corpora"
WORK = HERE / "work"
RESULTS = HERE / "results"
NZ = ROOT / "target" / "release" / "neurozip"

# Two NeuroZip configurations are measured, both at their compiled-in
# defaults:
#   "neurozip"      the bit-level mixing model with a match model (default)
#   "neurozip ctx"  the earlier byte-level context model, kept as the baseline
# The mixing model's constants were tuned only on the development set
# (bench/dev/, see dev.lock.json), never on the files measured here.
#
# Each entry: display name -> (compress argv, decompress argv). The
# placeholders are {src} (input), {out} (compressed), {dec} (restored). An
# argv ending in ">" + placeholder writes the tool's stdout to that file.
#
# The last three are the other context-modelling compressors, the class
# NeuroZip belongs to: PPMd (via 7-Zip, at its "ultra" preset), lpaq1 at
# level 6 (195 MB of tables, close to NeuroZip's memory) and zpaq at
# its strongest method. 7-Zip and zpaq write archives that also record the
# file name and attributes; those bytes are counted, as for every tool.
LPAQ = HERE / "tools" / "lpaq1"
SEVENZIP = ["7zz", "a", "-t7z", "-mx=9", "-m0=PPMd", "-mmt=1", "-y", "-bso0", "-bsp0"]

# Test sets: name -> (directory, lock file).
#   "final"  assembled after the current model and its constants were fixed;
#            the only set the current model has not been seen on before.
#   "fresh"  the second set; the previous model was measured on it once, and
#            its results were seen before the current model was built.
#   "first"  the original set, which informed the decision to add a match
#            model.
SETS = {
    "final": (HERE / "corpora3", HERE / "corpora3.lock.json"),
    "fresh": (HERE / "corpora2", HERE / "corpora2.lock.json"),
    "first": (HERE / "corpora", HERE / "corpora.lock.json"),
}

COMPRESSORS = {
    "ppmd (7z)": (SEVENZIP + ["{out}", "{src}"],
                  ["7zz", "e", "-so", "-bso0", "-bsp0", "{out}", ">{dec}"]),
    "lpaq1 6": ([str(LPAQ), "6", "{src}", "{out}"],
                [str(LPAQ), "d", "{out}", "{dec}"]),
    "zpaq -m5": (["zpaq", "a", "{out}", "{src}", "-m5", "-t1"],
                 ["zpaq", "x", "{out}", "{src}", "-to", "{dec}", "-f", "-t1"]),
    "neurozip": ([str(NZ), "compress", "{src}", "-o", "{out}", "-f"],
                 [str(NZ), "decompress", "{out}", "-o", "{dec}", "-f"]),
    "neurozip ctx": ([str(NZ), "compress", "{src}", "-o", "{out}", "-f", "--model", "ctx"],
                     [str(NZ), "decompress", "{out}", "-o", "{dec}", "-f"]),
    "zstd -3": (["zstd", "-3", "-T1", "-q", "-f", "{src}", "-o", "{out}"],
                ["zstd", "-d", "-q", "-f", "{out}", "-o", "{dec}"]),
    "zstd -19": (["zstd", "-19", "-T1", "-q", "-f", "{src}", "-o", "{out}"],
                 ["zstd", "-d", "-q", "-f", "{out}", "-o", "{dec}"]),
    "brotli -q 11": (["brotli", "-q", "11", "-f", "{src}", "-o", "{out}"],
                     ["brotli", "-d", "-f", "{out}", "-o", "{dec}"]),
    "xz -9": (["xz", "-9", "-T1", "-c", "{src}", ">{out}"],
              ["xz", "-d", "-c", "{out}", ">{dec}"]),
}

# name -> extra arguments to `neurozip ablate`
ABLATIONS = {
    "mix_full": [],
    "mix_no_match": ["--mix-off", "match"],
    "mix_no_apm": ["--mix-off", "apm"],
    "mix_no_word": ["--mix-off", "word"],
    "mix_no_x86": ["--mix-off", "x86"],
    "mix_frozen_mixer": ["--mix-off", "mixer"],
    "mix_no_fallback": ["--no-fallback"],
    "mix_independent_blocks": ["--independent-blocks"],
    "ctx_orders": ["--model", "ctx", "--orders", "0,1,2,3,4,5,6"],
    "ctx_freeze_10": ["--model", "ctx", "--freeze-after", "10"],
    "stored_only": ["--model", "stored"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_measured(argv):
    """Run one process; return (wall seconds, cpu seconds, peak RSS bytes)."""
    stdout = subprocess.DEVNULL
    if argv[-1].startswith(">"):
        stdout = open(argv[-1][1:], "wb")
        argv = argv[:-1]
    try:
        start = time.perf_counter()
        proc = subprocess.Popen(argv, stdout=stdout, stderr=subprocess.PIPE)
        _, status, usage = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - start
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        proc.returncode = os.waitstatus_to_exitcode(status)
    finally:
        if stdout is not subprocess.DEVNULL:
            stdout.close()
    if proc.returncode != 0:
        raise RuntimeError("%s failed (%d): %s" % (argv[0], proc.returncode, err.strip()))
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return wall, usage.ru_utime + usage.ru_stime, rss


def measure(argv, reps, fresh_output=None):
    """Time `argv` `reps` times. `fresh_output` is removed before each run,
    for tools that would otherwise add to an existing archive."""
    runs = []
    for _ in range(reps):
        if fresh_output is not None and fresh_output.exists():
            fresh_output.unlink()
        runs.append(run_measured(argv))
    return {
        "wall_s": statistics.median(r[0] for r in runs),
        "wall_s_min": min(r[0] for r in runs),
        "wall_s_max": max(r[0] for r in runs),
        "cpu_s": statistics.median(r[1] for r in runs),
        "peak_rss_bytes": max(r[2] for r in runs),
    }


def bench_file(directory, name, meta, reps):
    src = directory / name
    rows = []
    for comp, (c_argv, d_argv) in COMPRESSORS.items():
        # 7-Zip and zpaq choose behaviour from the archive's extension.
        ext = {"ppmd (7z)": ".7z", "zpaq -m5": ".zpaq"}.get(comp, ".out")
        out = WORK / (name + ext)
        dec = WORK / (name + ".dec")
        fill = lambda argv: [a.format(src=src, out=out, dec=dec) for a in argv]
        enc = measure(fill(c_argv), reps, fresh_output=out)
        size = out.stat().st_size
        dcm = measure(fill(d_argv), reps, fresh_output=dec)
        if sha256(dec) != meta["sha256"]:
            raise RuntimeError("%s did not reproduce %s" % (comp, name))
        out.unlink()
        dec.unlink()
        rows.append({
            "file": name,
            "category": meta["category"],
            "compressor": comp,
            "original_bytes": meta["bytes"],
            "compressed_bytes": size,
            "bits_per_byte": size * 8 / meta["bytes"],
            "encode": enc,
            "decode": dcm,
            "encode_mb_s": meta["bytes"] / 1e6 / enc["wall_s"],
            "decode_mb_s": meta["bytes"] / 1e6 / dcm["wall_s"],
            "verified_sha256": True,
        })
        print("%-26s %-13s %9d  %6.3f bpb  enc %8.2f MB/s  dec %8.2f MB/s  rss %6.1f/%6.1f MiB" % (
            name, comp, size, rows[-1]["bits_per_byte"], rows[-1]["encode_mb_s"],
            rows[-1]["decode_mb_s"], enc["peak_rss_bytes"] / 2**20, dcm["peak_rss_bytes"] / 2**20),
            file=sys.stderr)
    return rows


def ablate(job):
    directory, name, variant, extra = job
    out = subprocess.run([str(NZ), "ablate", str(directory / name)] + extra,
                         check=True, capture_output=True, text=True).stdout
    rows = [json.loads(line) for line in out.splitlines()]
    for row in rows:
        row["variant"] = variant
    return rows


def tool_version(argv):
    try:
        out = subprocess.run(argv, capture_output=True, text=True)
        return (out.stdout + out.stderr).strip().splitlines()[0]
    except (OSError, IndexError):
        return "unknown"


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions per run")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--sets", default="final,fresh,first", help="test sets to run, comma separated")
    ap.add_argument("--only", default="", help="only files whose name contains this (for testing)")
    ap.add_argument("--output", default="results.json", help="file name under results/")
    args = ap.parse_args()

    locks = {}
    for set_name, (directory, lock_path) in SETS.items():
        locks[set_name] = json.loads(lock_path.read_text())
        for name, meta in locks[set_name].items():
            if sha256(directory / name) != meta["sha256"]:
                sys.exit("%s does not match %s; run fetch_corpora.py" % (name, lock_path.name))
    for tool in ["zstd", "brotli", "xz", "7zz", "zpaq"]:
        if not shutil.which(tool):
            sys.exit("missing comparison tool: " + tool)
    if not LPAQ.exists():
        sys.exit("build lpaq1 first: sh bench/build_tools.sh")
    if not NZ.exists():
        sys.exit("build first: cargo build --release")

    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    sets = {}
    for set_name in args.sets.split(","):
        directory = SETS[set_name][0]
        print("== test set: %s ==" % set_name, file=sys.stderr)
        runs = []
        for name, meta in locks[set_name].items():
            if args.only and args.only not in name:
                continue
            runs.extend(bench_file(directory, name, meta, args.reps))
        sets[set_name] = {"corpora": locks[set_name], "runs": runs}

    # Ablations are measured on the final set only.
    lock = locks["final"]
    fresh_dir = SETS["final"][0]
    ablations = []
    if not args.skip_ablations:
        jobs = [(fresh_dir, name, variant, extra)
                for name in lock for variant, extra in ABLATIONS.items()]
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2)) as pool:
            for rows in pool.map(ablate, jobs):
                ablations.extend(rows)
        categories = {name: meta["category"] for name, meta in lock.items()}
        for row in ablations:
            row["category"] = categories[row["file"]]
        print("ablations: %d rows" % len(ablations), file=sys.stderr)

    results = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "logical_cpus": os.cpu_count(),
            "memory_bytes": int(sysctl("hw.memsize") or 0),
            "os": "%s %s" % (platform.system(), platform.mac_ver()[0] or platform.release()),
            "arch": platform.machine(),
        },
        "tools": {
            "neurozip": tool_version([str(NZ), "--version"]),
            "neurozip_config": "mix model: hash bits 18, match window 2^22, block 256 KiB; "
                               "ctx model: order 4, hash bits 20, block 256 KiB",
            "zstd": tool_version(["zstd", "--version"]),
            "brotli": tool_version(["brotli", "--version"]),
            "xz": tool_version(["xz", "--version"]),
            "7zz": next((line for line in subprocess.run(["7zz"], capture_output=True, text=True)
                         .stdout.splitlines() if line.strip()), "unknown"),
            "zpaq": "zpaq 7.15, method 5",
            "lpaq1": "lpaq1 (Matt Mahoney, 2007), level 6, built from source with c++ -O2",
            "rustc": tool_version(["rustc", "--version"]),
        },
        "method": {
            "repetitions": args.reps,
            "timing": "median wall time of whole CLI process, file to file, one run at a time",
            "memory": "peak resident set size of the process (wait4)",
            "cpu": "user + system CPU time of the process (wait4)",
            "power": "not measured (needs root-only powermetrics on macOS)",
            "size": "size of the output file, all headers and checksums included",
        },
        "sets": sets,
        "ablations": ablations,
    }
    path = RESULTS / args.output
    path.write_text(json.dumps(results, indent=1) + "\n")
    shutil.rmtree(WORK, ignore_errors=True)
    print("wrote", path, file=sys.stderr)


if __name__ == "__main__":
    main()
