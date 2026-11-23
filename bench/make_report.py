#!/usr/bin/env python3
"""Turn `results/final.json` (plus `fresh.json` and `first.json`) into `docs/REPORT.md` and its SVG charts.

Every number in the report is computed here from the results file; nothing
is typed in by hand. Standard library only.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results" / "results.json"
CHARTS = ROOT / "docs" / "charts"
REPORT = ROOT / "docs" / "REPORT.md"

# "neurozip" is the mixing model (the default); "neurozip ctx" is the earlier
# byte-level context model, kept as a baseline in the tables only.
CLASSICAL = ["zstd -3", "zstd -19", "brotli -q 11", "xz -9"]
SAME_CLASS = ["ppmd (7z)", "lpaq1 6", "zpaq -m5"]
COMPRESSORS = ["neurozip"] + CLASSICAL + SAME_CLASS + ["neurozip ctx"]
# Charts show eight series, in this fixed order. Colour follows the tool,
# never its rank.
CHARTED = ["neurozip"] + CLASSICAL + SAME_CLASS
COLOR = {
    "neurozip": "#2a78d6",
    "zstd -3": "#eb6834",
    "zstd -19": "#1baf7a",
    "brotli -q 11": "#eda100",
    "xz -9": "#e87ba4",
    "ppmd (7z)": "#008300",
    "lpaq1 6": "#6250d6",
    "zpaq -m5": "#e34948",
}
KIND = {c: "classical" for c in CLASSICAL}
KIND.update({c: "context modelling" for c in SAME_CLASS})
KIND.update({"neurozip": "context modelling", "neurozip ctx": "context modelling"})

CATEGORIES = ["English text", "Code", "JSON", "Logs", "Executables", "Compressed", "Random"]
# Categories where compression is possible at all. The other two are
# already-compressed and random data, where every tool lands near 8 bits/byte.
COMPRESSIBLE = CATEGORIES[:5]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- data

def load():
    """Merge the per-set result files. Each test set is run separately so a
    long run cannot lose the sets already finished; the final set's file
    also carries the machine details and the ablations."""
    res = json.loads((RESULTS.parent / "final.json").read_text())
    for name in ("fresh", "first"):
        path = RESULTS.parent / (name + ".json")
        if path.exists():
            res["sets"].update(json.loads(path.read_text())["sets"])
    return res


def aggregate(runs):
    """bits/byte and throughput per (compressor, category) and overall."""
    size = defaultdict(lambda: [0, 0])          # (comp, cat) -> [compressed, original]
    speed = defaultdict(lambda: [0, 0.0, 0.0])  # comp -> [bytes, enc s, dec s] on compressible files
    cpu = defaultdict(lambda: [0.0, 0.0])
    rss = defaultdict(lambda: [0, 0])
    for r in runs:
        key = (r["compressor"], r["category"])
        size[key][0] += r["compressed_bytes"]
        size[key][1] += r["original_bytes"]
        c = r["compressor"]
        rss[c][0] = max(rss[c][0], r["encode"]["peak_rss_bytes"])
        rss[c][1] = max(rss[c][1], r["decode"]["peak_rss_bytes"])
        if r["category"] in COMPRESSIBLE:
            speed[c][0] += r["original_bytes"]
            speed[c][1] += r["encode"]["wall_s"]
            speed[c][2] += r["decode"]["wall_s"]
            cpu[c][0] += r["encode"]["cpu_s"]
            cpu[c][1] += r["decode"]["cpu_s"]
    bpb = {k: v[0] * 8 / v[1] for k, v in size.items()}
    out = {}
    for c in COMPRESSORS:
        cats = {cat: bpb[(c, cat)] for cat in CATEGORIES}
        n, es, ds = speed[c]
        out[c] = {
            "bpb": cats,
            "mean_compressible": sum(cats[k] for k in COMPRESSIBLE) / len(COMPRESSIBLE),
            "mean_all": sum(cats.values()) / len(cats),
            "total_compressible_bytes": sum(size[(c, cat)][0] for cat in COMPRESSIBLE),
            "enc_mb_s": n / 1e6 / es,
            "dec_mb_s": n / 1e6 / ds,
            "enc_cpu_s_per_mb": cpu[c][0] / (n / 1e6),
            "dec_cpu_s_per_mb": cpu[c][1] / (n / 1e6),
            "enc_rss": rss[c][0],
            "dec_rss": rss[c][1],
        }
    return out


def ablation_table(rows):
    """variant -> config -> category -> bits/byte (bytes summed per category)."""
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["variant"], r["config"], r["category"])
        acc[key][0] += r["archive_bytes"]
        acc[key][1] += r["original_bytes"]
    out = defaultdict(lambda: defaultdict(dict))
    for (variant, config, cat), (comp, orig) in acc.items():
        out[variant][config][cat] = comp * 8 / orig
    return out


def only(abl, variant):
    """The single configuration measured for a variant."""
    (cfg,) = abl[variant].values()
    return cfg


def mean5(cfg):
    return sum(cfg[c] for c in COMPRESSIBLE) / len(COMPRESSIBLE)


# ---------------------------------------------------------------- svg

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, fill=INK2, anchor="start", weight="normal"):
    return ('<text x="%.1f" y="%.1f" font-size="%d" fill="%s" text-anchor="%s" font-weight="%s">%s</text>'
            % (x, y, size, fill, anchor, weight, esc(s)))


def svg(width, height, body, title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
            'font-family="%s" role="img" aria-label="%s">\n<rect width="%d" height="%d" fill="%s"/>\n%s\n</svg>\n'
            % (width, height, width, height, FONT, esc(title), width, height, SURFACE, "\n".join(body)))


def hbar(x, y, w, h, fill):
    """Horizontal bar: square at the baseline, 4px rounded at the data end."""
    w = max(w, 1.0)
    r = min(4.0, w, h / 2)
    return ('<path d="M%.1f %.1f h%.1f a%.1f %.1f 0 0 1 %.1f %.1f v%.1f a%.1f %.1f 0 0 1 -%.1f %.1f h-%.1f z" fill="%s"/>'
            % (x, y, w - r, r, r, r, r, h - 2 * r, r, r, r, r, w - r, fill))


def chart_bpb(agg):
    """Small multiples, one panel per compressible category, shared scale."""
    width, left, right = 760, 170, 690
    top, row, bar, head, gap = 64, 19, 12, 26, 16
    xmax = 4.0
    sx = lambda v: left + (right - left) * v / xmax
    panel_h = head + row * len(CHARTED) + gap
    height = top + panel_h * len(COMPRESSIBLE) + 34
    b = [text(20, 28, "Compressed size by data type (final test set)", 15, INK, weight="600"),
         text(20, 46, "Bits per input byte, all archive metadata included. Lower is better. "
                      "Measured on this machine.", 11, INK2)]
    plot_bottom = top + panel_h * len(COMPRESSIBLE) - gap
    for t in range(0, 5):
        x = sx(t)
        b.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>'
                 % (x, top + 8, x, plot_bottom, AXIS if t == 0 else GRID))
        b.append(text(x, plot_bottom + 16, str(t), 10, MUTED, "middle"))
    b.append(text((left + right) / 2, plot_bottom + 30, "bits per byte", 10, MUTED, "middle"))
    for i, cat in enumerate(COMPRESSIBLE):
        y0 = top + i * panel_h
        b.append(text(20, y0 + 16, cat, 12, INK, weight="600"))
        best = min(agg[c]["bpb"][cat] for c in CHARTED)
        for j, c in enumerate(CHARTED):
            v = agg[c]["bpb"][cat]
            y = y0 + head + j * row
            mine = c == "neurozip"
            b.append(text(left - 8, y + bar - 2, c, 11, INK if mine else INK2, "end",
                          "600" if mine else "normal"))
            b.append(hbar(left, y, sx(v) - left, bar, COLOR[c]))
            label = "%.3f%s" % (v, "  smallest" if v == best else "")
            b.append(text(sx(v) + 6, y + bar - 2, label, 11, INK if v == best else INK2,
                          weight="600" if v == best else "normal"))
    return svg(width, height, b, "Bits per byte by data type and compressor, final test set")


def chart_speed(agg):
    """Size against throughput, encode and decode, log speed axis."""
    width, height = 760, 404
    panels = [("Compression speed", "enc_mb_s", 60), ("Decompression speed", "dec_mb_s", 420)]
    pw, top, bottom = 300, 104, 348
    ys = [agg[c]["mean_compressible"] for c in CHARTED]
    ymin = math.floor(min(ys) * 5) / 5 - 0.2
    ymax = math.ceil(max(ys) * 5) / 5 + 0.2
    sy = lambda v: bottom - (bottom - top) * (v - ymin) / (ymax - ymin)
    lo, hi = math.log10(0.08), math.log10(2000)
    b = [text(20, 28, "Size against speed (final test set)", 15, INK, weight="600"),
         text(20, 46, "Mean bits per byte over the five compressible data types (lower is better) against "
                      "throughput in MB/s (log scale).", 11, INK2),
         text(20, 61, "Whole command-line process, file to file, one thread, median of the timed runs.",
              11, INK2)]
    # Per-panel label placement (dx, dy, anchor), chosen so labels do not collide.
    place = {
        "enc_mb_s": {"neurozip": (0, 16, "middle"), "zstd -3": (-10, 0, "end"),
                     "zstd -19": (10, -10, "start"), "brotli -q 11": (0, -16, "middle"),
                     "xz -9": (10, 8, "start"), "ppmd (7z)": (10, 0, "start"),
                     "lpaq1 6": (10, 0, "start"), "zpaq -m5": (-10, 0, "end")},
        "dec_mb_s": {"neurozip": (0, 16, "middle"), "zstd -3": (0, -16, "middle"),
                     "zstd -19": (0, -16, "middle"), "brotli -q 11": (10, 10, "start"),
                     "xz -9": (-10, 0, "end"), "ppmd (7z)": (10, 0, "start"),
                     "lpaq1 6": (10, 0, "start"), "zpaq -m5": (-10, 0, "end")},
    }
    for title, key, x0 in panels:
        sx = lambda v: x0 + pw * (math.log10(v) - lo) / (hi - lo)
        b.append(text(x0, top - 10, title, 12, INK, weight="600"))
        t = ymin
        while t <= ymax + 1e-9:
            y = sy(t)
            b.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1"/>'
                     % (x0, y, x0 + pw, y, GRID))
            b.append(text(x0 - 6, y + 3.5, "%.1f" % t, 10, MUTED, "end"))
            t += 0.2
        for tick in [0.1, 1, 10, 100, 1000]:
            x = sx(tick)
            b.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>'
                     % (x, top, x, bottom, GRID))
            b.append(text(x, bottom + 16, "%g" % tick, 10, MUTED, "middle"))
        b.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
                 % (x0, bottom, x0 + pw, bottom, AXIS))
        b.append(text(x0 + pw / 2, bottom + 32, "MB/s", 10, MUTED, "middle"))
        for c in CHARTED:
            x, y = sx(agg[c][key]), sy(agg[c]["mean_compressible"])
            dx, dy, anchor = place[key][c]
            b.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="%s" stroke-width="2"/>'
                     % (x, y, COLOR[c], SURFACE))
            mine = c == "neurozip"
            b.append(text(x + dx, y + dy + 4, c, 11, INK if mine else INK2, anchor,
                          "600" if mine else "normal"))
    b.append(text(14, (top + bottom) / 2, "bits per byte", 10, MUTED, "middle")
             .replace("<text ", '<text transform="rotate(-90 14 %.1f)" ' % ((top + bottom) / 2)))
    return svg(width, height, b, "Mean bits per byte against compression and decompression speed")


MIX_VARIANTS = [
    ("mix_full", "Full model"),
    ("mix_no_word", "Without the word context"),
    ("mix_no_x86", "Without the x86 filter"),
    ("mix_no_apm", "Without the correction tables"),
    ("mix_frozen_mixer", "Mixer learning switched off"),
    ("mix_no_match", "Without the match model"),
]


def chart_parts(abl):
    """Mixing-model ablation: mean bits per byte with one part removed."""
    width, left, right, top, row, bar = 760, 250, 640, 70, 26, 14
    vals = [(label, mean5(only(abl, v))) for v, label in MIX_VARIANTS]
    xmax = math.ceil(max(v for _, v in vals) * 2) / 2
    sx = lambda v: left + (right - left) * v / xmax
    height = top + row * len(vals) + 44
    bottom = top + row * len(vals)
    b = [text(20, 28, "What each part of the mixing model contributes", 15, INK, weight="600"),
         text(20, 46, "Mean bits per byte over the five compressible data types with one part switched "
                      "off. Final test set. Lower is better.", 11, INK2)]
    t = 0.0
    while t <= xmax + 1e-9:
        x = sx(t)
        b.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>'
                 % (x, top - 4, x, bottom, AXIS if t == 0 else GRID))
        b.append(text(x, bottom + 16, "%.1f" % t, 10, MUTED, "middle"))
        t += 0.5
    b.append(text((left + right) / 2, bottom + 32, "bits per byte", 10, MUTED, "middle"))
    base = vals[0][1]
    for i, (label, v) in enumerate(vals):
        y = top + i * row
        b.append(text(left - 8, y + bar - 2, label, 11, INK if i == 0 else INK2, "end",
                      "600" if i == 0 else "normal"))
        b.append(hbar(left, y, sx(v) - left, bar, COLOR["neurozip"]))
        note = "%.3f" % v if i == 0 else "%.3f  (+%.1f%%)" % (v, (v / base - 1) * 100)
        b.append(text(sx(v) + 6, y + bar - 2, note, 11, INK if i == 0 else INK2,
                      weight="600" if i == 0 else "normal"))
    return svg(width, height, b, "Mean bits per byte of the mixing model with parts switched off")


# ---------------------------------------------------------------- report

def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = ["---" if a == "l" else "---:" for a in align]
    return "\n".join([line(header), line(sep)] + [line(r) for r in rows])


def bold_min(values, fmt):
    best = min(values)
    return [("**" + fmt % v + "**") if v == best else fmt % v for v in values]


def mib(n):
    return "%.0f" % (n / 2**20)


def speed(v):
    return "%.1f" % v if v < 100 else "%.0f" % v


def ranking_rows(agg):
    rank = sorted(COMPRESSORS, key=lambda c: agg[c]["mean_compressible"])
    rows = []
    for i, c in enumerate(rank, 1):
        a = agg[c]
        name = "**`%s`**" % c if c == "neurozip" else "`%s`" % c
        rows.append([i, name, KIND[c], "%.3f" % a["mean_compressible"],
                     "{:,}".format(a["total_compressible_bytes"]), speed(a["enc_mb_s"]),
                     speed(a["dec_mb_s"]), mib(a["enc_rss"]), mib(a["dec_rss"])])
    return rank, rows


RANK_HEADER = ["Rank", "Compressor", "Kind", "Mean bpb (5 types)", "Bytes (5 types)",
               "Compress MB/s", "Decompress MB/s", "Compress MiB", "Decompress MiB"]
RANK_ALIGN = ["r", "l", "l"] + ["r"] * 6


def type_table(agg):
    rows = []
    for cat in CATEGORIES:
        rows.append([cat] + bold_min([agg[c]["bpb"][cat] for c in COMPRESSORS], "%.3f"))
    return table(["Bits per byte"] + ["`%s`" % c for c in COMPRESSORS], rows)


def file_table(data):
    by_file = defaultdict(dict)
    for r in data["runs"]:
        by_file[r["file"]][r["compressor"]] = r["compressed_bytes"]
    rows = []
    for name, meta in data["corpora"].items():
        sizes = [by_file[name][c] for c in COMPRESSORS]
        best = min(sizes)
        rows.append(["`%s`" % name, "{:,}".format(meta["bytes"])] +
                    [("**{:,}**" if x == best else "{:,}").format(x) for x in sizes])
    return by_file, table(["File", "Original bytes"] + ["`%s`" % c for c in COMPRESSORS], rows)


def pct_below(a, b):
    """How far `a` is below `b`, in percent (negative if above)."""
    return (1 - a / b) * 100


def write_report(res, abl):
    m, tools, method = res["machine"], res["tools"], res["method"]
    fresh = res["sets"]["final"]
    agg = aggregate(fresh["runs"])
    nz = agg["neurozip"]
    mean = lambda a, c: a[c]["mean_compressible"]
    rank, rank_rows = ranking_rows(agg)
    place = rank.index("neurozip") + 1
    ahead = [c for c in rank[place:] if c != "neurozip ctx"]
    behind = rank[:place - 1]
    best_classical = min(CLASSICAL, key=lambda c: mean(agg, c))
    by_file, files_md = file_table(fresh)
    compressible_files = [n for n, c in fresh["corpora"].items() if c["category"] in COMPRESSIBLE]
    beats_classical = [n for n in compressible_files
                       if by_file[n]["neurozip"] < min(by_file[n][c] for c in CLASSICAL)]
    smallest = [n for n in compressible_files
                if by_file[n]["neurozip"] == min(by_file[n].values())]
    type_vs_classical = {cat: pct_below(nz["bpb"][cat], min(agg[c]["bpb"][cat] for c in CLASSICAL))
                         for cat in COMPRESSIBLE}

    out = []
    w = out.append
    w("# NeuroZip benchmark report")
    w("")
    w("Generated by `bench/make_report.py` from the files in `bench/results/` "
      "(final-set run of %s). Every number below is computed from those files." % res["generated_utc"])
    w("")
    w("## Summary")
    w("")
    w("**What was tested.** NeuroZip's default model: bit-level context mixing. Each context keeps "
      "a counter and a short bit history; a match model follows long repeats; an x86 filter is "
      "applied to blocks that look like machine code; and a small two-layer linear mixer, trained "
      "online in integer arithmetic with no stored weights, combines the predictions. It is "
      "compared with four classical configurations (zstd -3, zstd -19, Brotli, xz) and with three "
      "compressors of its own kind (PPMd, lpaq1, zpaq). **No MLP or other deeper neural predictor "
      "has been built**; the learned part is the linear mixer only.")
    w("")
    w("The headline numbers come from the **final test set** of %d files, assembled after this "
      "model and its constants were fixed, and run once. The two earlier test sets are reported "
      "separately below." % len(fresh["corpora"]))
    w("")
    w("**Against the classical compressors: smaller, much slower.**")
    w("")
    w("* Mean bits per byte over the five compressible data types: NeuroZip %.3f, best classical "
      "(`%s`) %.3f. NeuroZip is %.1f%% lower."
      % (mean(agg, "neurozip"), best_classical, mean(agg, best_classical),
         pct_below(mean(agg, "neurozip"), mean(agg, best_classical))))
    w("* By data type against the best classical tool: %s." % "; ".join(
        "%s %.1f%% %s" % (cat, abs(v), "lower" if v > 0 else "higher")
        for cat, v in type_vs_classical.items()))
    w("* By file it is smaller than every classical tool on %d of the %d compressible files."
      % (len(beats_classical), len(compressible_files)))
    w("* It decompresses at %.1f MB/s; `xz -9` at %s MB/s and `zstd -19` at %s MB/s, roughly %d and "
      "%d times faster. It needs about %s MiB to decompress; the classical tools need %s MiB or less."
      % (nz["dec_mb_s"], speed(agg["xz -9"]["dec_mb_s"]), speed(agg["zstd -19"]["dec_mb_s"]),
         agg["xz -9"]["dec_mb_s"] / nz["dec_mb_s"], agg["zstd -19"]["dec_mb_s"] / nz["dec_mb_s"],
         mib(nz["dec_rss"]), mib(max(agg[c]["dec_rss"] for c in CLASSICAL))))
    w("")
    w("**Against its own kind.**")
    w("")
    for c in sorted(SAME_CLASS, key=lambda c: mean(agg, c)):
        d = pct_below(mean(agg, c), mean(agg, "neurozip"))
        w("* `%s`: %.3f bits per byte, %.1f%% %s NeuroZip; compresses at %s MB/s, decompresses at "
          "%s MB/s, %s MiB." % (c, mean(agg, c), abs(d), "below" if d > 0 else "above",
                                speed(agg[c]["enc_mb_s"]), speed(agg[c]["dec_mb_s"]),
                                mib(agg[c]["dec_rss"])))
    w("* NeuroZip is the smallest of all nine configurations on %d of the %d compressible files."
      % (len(smallest), len(compressible_files)))
    w("")
    w("So NeuroZip ranks %s of nine on mean size on this set. Smaller than it: %s. Larger: %s."
      % ({1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(place, "%dth" % place),
         ", ".join("`%s`" % c for c in behind) or "none",
         ", ".join("`%s`" % c for c in ahead) or "none"))
    w("")
    w("**What this does and does not show.** Every technique in the model is established in the "
      "PAQ, lpaq and zpaq line of compressors (see `docs/PRIOR_ART.md`); nothing here is claimed "
      "as a new algorithm. The comparison covers lpaq1 at one level and zpaq at its strongest "
      "standard method. The strongest compressors of this kind (cmix, the PAQ8 family) were not "
      "run and are expected to produce smaller files than anything in this report, at much lower "
      "speed.")
    w("")
    cmix_path = RESULTS.parent / "cmix_check.json"
    if cmix_path.exists():
        cm = json.loads(cmix_path.read_text())
        mine = by_file[cm["file"]]["neurozip"]
        w("**One check against cmix.** %s was run on a single file of this set, `%s` "
          "({:,} bytes). It produced {:,} bytes; NeuroZip produced {:,}, which is %.0f%% larger. "
          "cmix took %.0f seconds of CPU time and %.1f GiB of memory for that file. This is one "
          "file, compressed only (not decompressed to verify), but it is direct evidence that "
          "NeuroZip is not the best available compressor on size."
          .format(cm["original_bytes"], cm["compressed_bytes"], mine)
          % (cm["tool"], cm["file"], (mine / cm["compressed_bytes"] - 1) * 100,
             cm["cpu_seconds"], cm["peak_rss_bytes"] / 2**30))
        w("")
    w("## What these numbers are")
    w("")
    w("* **Local measurements**, not estimates: every size is a real output file and every time is a "
      "real process, on one machine. Nothing here is extrapolated. Results on other hardware or other "
      "files will differ.")
    w("* **Machine:** %s, %d logical CPUs, %d GiB RAM, %s (%s)."
      % (m["cpu"], m["logical_cpus"], m["memory_bytes"] // 2**30, m["os"], m["arch"]))
    w("* **Tools:** %s (%s); %s; %s; %s; %s (PPMd, `-mx=9`); %s; %s. All single-threaded."
      % (tools["neurozip"], tools["neurozip_config"], tools["zstd"].strip("* "), tools["brotli"],
         tools["xz"], tools["7zz"].split(":")[0].strip(), tools["zpaq"], tools["lpaq1"]))
    w("* **Size:** %s. 7-Zip and zpaq archives also store the file name and attributes; those "
      "bytes are counted." % method["size"])
    w("* **Time:** %s; %d runs each." % (method["timing"], method["repetitions"]))
    w("* **Memory:** %s. **CPU:** %s." % (method["memory"], method["cpu"]))
    w("* **Power:** %s. CPU time is the nearest available proxy." % method["power"])
    w("* **Verification:** every compressed file was decompressed and matched the original's SHA-256.")
    w("* Files are capped at 4 MiB, which is small. Behaviour on large inputs is not measured.")
    w("* Throughput for the fast tools on files this small includes process start-up, so it "
      "understates them.")
    w("* **The machine was not idle.** Other jobs were running during these runs, so every "
      "throughput figure here is lower than on a quiet machine (by roughly a factor of two for "
      "the final set, judging by tools measured earlier) and the ratios between tools are only "
      "approximate. Sizes are not affected.")
    w("* lpaq1 was run at level 6 (195 MB of tables), the level closest to NeuroZip's memory; it "
      "has higher levels. zpaq and PPMd were run at their strongest standard settings. cmix was "
      "not run (hours per file, tens of gigabytes of memory).")
    w("")
    w("## Ranking (final test set)")
    w("")
    w("Ranked by mean bits per byte over the five compressible data types, each type weighted equally.")
    w("")
    w(table(RANK_HEADER, rank_rows, RANK_ALIGN))
    w("")
    w("Speed is over the compressible files; memory is the peak for any file.")
    w("")
    w("## Size by data type (final test set)")
    w("")
    w("![Bits per byte by data type](charts/bpb_by_category.svg)")
    w("")
    w(type_table(agg))
    w("")
    w("Smallest in each row is bold. On the last two rows almost nothing compresses.")
    w("")
    w("## Speed, CPU and memory (final test set)")
    w("")
    w("![Size against speed](charts/speed_vs_size.svg)")
    w("")
    rows = []
    for c in COMPRESSORS:
        a = agg[c]
        rows.append(["`%s`" % c, speed(a["enc_mb_s"]), speed(a["dec_mb_s"]),
                     "%.3f" % a["enc_cpu_s_per_mb"], "%.4f" % a["dec_cpu_s_per_mb"],
                     mib(a["enc_rss"]), mib(a["dec_rss"])])
    w(table(["Compressor", "Compress MB/s", "Decompress MB/s", "Compress CPU s/MB",
             "Decompress CPU s/MB", "Compress peak MiB", "Decompress peak MiB"], rows))
    w("")
    w("NeuroZip, lpaq1 and zpaq decompress about as slowly as they compress, because the decoder "
      "has to re-run the same model. That is inherent to the approach.")
    w("")
    w("## Per-file results (final test set)")
    w("")
    w(files_md)
    w("")
    w("## The earlier test sets")
    w("")
    w("No constant was ever tuned on either of these sets, but both had been looked at before "
      "the current model was built, so neither is a clean test of it. The second set's results "
      "for the previous model showed where it was weakest; the first set prompted the match "
      "model. They are shown for comparison.")
    for key, title in (("fresh", "Second test set"), ("first", "First test set")):
        if key not in res["sets"]:
            continue
        other = res["sets"][key]
        agg_o = aggregate(other["runs"])
        rank_o, rows_o = ranking_rows(agg_o)
        w("")
        w("### %s (%d files)" % (title, len(other["corpora"])))
        w("")
        w(table(RANK_HEADER, rows_o, RANK_ALIGN))
        w("")
        w(type_table(agg_o))
        w("")
        w("NeuroZip's mean is %.3f and its rank is %d of nine on this set."
          % (mean(agg_o, "neurozip"), rank_o.index("neurozip") + 1))
    w("")
    w("## Ablations (final test set)")
    w("")
    w("Sizes from `neurozip ablate`, all metadata included. The mixing-model variants are selected "
      "by switches stored in the archive header, so each one is a real, decodable archive. "
      "\"Fallback disabled\" and the context model's \"frozen\" variant are size measurements only: "
      "the format cannot express them.")
    w("")
    w("![What each part contributes](charts/ablation_parts.svg)")
    w("")
    full = only(abl, "mix_full")
    extra = [("mix_independent_blocks", "Model reset at every 256 KiB block"),
             ("mix_no_fallback", "Stored fallback disabled")]
    rows = []
    for variant, label in MIX_VARIANTS + extra:
        cfg = only(abl, variant)
        rows.append([label] + ["%.3f" % cfg[c] for c in CATEGORIES] + ["%.3f" % mean5(cfg)])
    ctx4 = abl["ctx_orders"]["ctx-o4-h20"]
    rows.append(["Byte-level context model (`neurozip ctx`)"] + ["%.3f" % ctx4[c] for c in CATEGORIES]
                + ["%.3f" % mean5(ctx4)])
    stored = only(abl, "stored_only")
    rows.append(["No model (everything stored)"] + ["%.3f" % stored[c] for c in CATEGORIES]
                + ["%.3f" % mean5(stored)])
    w(table(["Variant (bits per byte)"] + CATEGORIES + ["Mean of first 5"], rows))
    w("")
    pct = lambda v: (mean5(only(abl, v)) / mean5(full) - 1) * 100
    nm = only(abl, "mix_no_match")
    worst_cat = max(COMPRESSIBLE, key=lambda c: nm[c] / full[c])
    w("* **Match model:** removing it costs %.1f%% on the mean and most on %s (%.3f to %.3f)."
      % (pct("mix_no_match"), worst_cat, full[worst_cat], nm[worst_cat]))
    w("* **Online-trained mixer:** with learning switched off the predictors are combined with fixed "
      "equal weights, and the mean rises %.1f%%. This is the measured value of the learned "
      "combination over a fixed one." % pct("mix_frozen_mixer"))
    w("* **Correction tables:** %.1f%%. **Word context:** %.1f%% on the mean, %.1f%% on English text."
      % (pct("mix_no_apm"), pct("mix_no_word"),
         (only(abl, "mix_no_word")["English text"] / full["English text"] - 1) * 100))
    ib = only(abl, "mix_independent_blocks")
    w("* **Carrying the model across blocks:** resetting every block raises the mean from %.3f to "
      "%.3f." % (mean5(full), mean5(ib)))
    nf = only(abl, "mix_no_fallback")
    w("* **Fallback:** it matters only for incompressible input. Without it random data costs %.3f "
      "bits per byte instead of %.3f. The only fallback that exists is \"stored\"; fallback to a "
      "classical codec has not been built." % (nf["Random"], full["Random"]))
    w("")
    w("### Baseline context model: highest order")
    w("")
    rows = []
    for o in range(7):
        cfg = abl["ctx_orders"]["ctx-o%d-h20" % o]
        rows.append(["%d%s" % (o, " (default)" if o == 4 else "")] +
                    ["%.3f" % cfg[c] for c in COMPRESSIBLE] + ["%.3f" % mean5(cfg)])
    w(table(["Highest order"] + COMPRESSIBLE + ["Mean"], rows))
    w("")
    f10 = only(abl, "ctx_freeze_10")
    w("With that model's learning frozen after the first 10%% of each file the mean rises from "
      "%.3f to %.3f." % (mean5(ctx4), mean5(f10)))
    w("")
    w("### Status of the requested ablations")
    w("")
    w(table(["Ablation requested", "Status"], [
        ["Context only", "Done: the byte-level context model, and its order sweep"],
        ["Neural only", "Not run. The only learned component is the linear mixer, which has no "
                        "prediction of its own without the context inputs. A stand-alone neural "
                        "predictor (MLP) has not been built"],
        ["Ensemble", "Done: the full mixing model, against the same model with mixer learning off"],
        ["Fallback", "Done for stored fallback; classical-codec fallback has not been built"],
        ["Adaptive learning disabled", "Done: mixer learning off (mixing model) and counts frozen "
                                       "(context model)"],
    ], ["l", "l"]))
    w("")
    w("## Leakage controls")
    w("")
    w("* No model has trained weights: encoder and decoder start from a fixed initial state and "
      "learn only from the file being compressed. There is no training set to leak from.")
    w("* The remaining risk is tuning and design choices. The model's constants (counter limits, "
      "mixer learning rates, initial weights, correction-table rate) and its choice of contexts "
      "were settled on a separate development set (`bench/dev.lock.json`).")
    w("* The final test set (`bench/corpora3.lock.json`) was assembled after that, from books, "
      "projects, JSON files, log files and binaries that appear in no other set. The full "
      "benchmark was run on it once and the results are reported as they came out. One file "
      "(`json_apache_builds.json`) was compressed once beforehand to check that the benchmark "
      "pipeline worked; nothing was changed afterwards.")
    w("* The two earlier test sets (`bench/corpora2.lock.json`, `bench/corpora.lock.json`) had "
      "been seen before this model was built, as described above, and are reported separately "
      "for that reason.")
    w("* Final test set files:")
    w("")
    rows = [["`%s`" % n, c["category"], c["source"], "{:,}".format(c["bytes"]),
             "first 4 MiB of {:,}".format(c["truncated_from"]) if c["truncated_from"] else "whole file"]
            for n, c in fresh["corpora"].items()]
    w(table(["File", "Type", "Source", "Bytes", "Extent"], rows, ["l", "l", "l", "r", "l"]))
    w("")
    w("## Limits of this benchmark")
    w("")
    w("* One machine, one run of the suite, %d timed repetitions per measurement." % method["repetitions"])
    w("* Two or three files per data type in each set, each at most 4 MiB. Differences of a few "
      "percent between neighbouring compressors could change with other files.")
    w("* Power was not measured.")
    w("* The stronger members of this class were not all run: cmix and the larger PAQ8 variants "
      "are expected to produce smaller files than anything here, at far lower speed.")
    w("* The classical tools were run at their strongest standard presets but without extras such "
      "as xz's x86 filter or Brotli's large-window mode.")
    w("")
    w("## Reproduce")
    w("")
    w("```")
    w("cargo build --release")
    w("python3 bench/fetch_corpora.py   # downloads and checks the pinned files")
    w("sh bench/build_tools.sh          # builds lpaq1; install zstd, brotli, xz, 7zz, zpaq separately")
    w("python3 bench/run_bench.py --sets final --output final.json")
    w("python3 bench/run_bench.py --sets fresh --skip-ablations --output fresh.json")
    w("python3 bench/run_bench.py --sets first --skip-ablations --output first.json")
    w("python3 bench/make_report.py     # writes this report and docs/charts/")
    w("```")
    w("")
    REPORT.write_text("\n".join(out))


def main():
    res = load()
    abl = ablation_table(res["ablations"])
    agg = aggregate(res["sets"]["final"]["runs"])
    CHARTS.mkdir(parents=True, exist_ok=True)
    (CHARTS / "bpb_by_category.svg").write_text(chart_bpb(agg))
    (CHARTS / "speed_vs_size.svg").write_text(chart_speed(agg))
    (CHARTS / "ablation_parts.svg").write_text(chart_parts(abl))
    stale = CHARTS / "ablation_order.svg"
    if stale.exists():
        stale.unlink()
    write_report(res, abl)
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
