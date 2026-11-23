#!/usr/bin/env python3
"""Build the held-out test corpora for the NeuroZip benchmark.

Every file comes from a fixed public URL (or a fixed seed) and is reduced to
at most CAP bytes. The result is checked against `corpora.lock.json`; if the
lock does not exist yet it is written. A mismatch means an upstream file
changed and the benchmark would no longer be comparable, so it is an error.

None of these files was looked at while the compressor was being written.
The files used for the informal first look (see README) are a separate
development set and are never part of this benchmark.

Standard library only. Downloads use `curl`.
"""

import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpora"
CACHE = HERE / "corpora" / ".downloads"
LOCK = HERE / "corpora.lock.json"
CAP = 4 * 1024 * 1024  # bytes kept per file

REDIS = "https://github.com/redis/redis/archive/refs/tags/7.2.4.tar.gz"
FLASK = "https://github.com/pallets/flask/archive/refs/tags/3.0.0.tar.gz"
RG_LINUX = ("https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/"
            "ripgrep-14.1.0-x86_64-unknown-linux-musl.tar.gz")
RG_WIN = ("https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/"
          "ripgrep-14.1.0-x86_64-pc-windows-msvc.zip")
JSON_BASE = "https://raw.githubusercontent.com/miloyip/nativejson-benchmark/master/data/"
LOG_BASE = "https://raw.githubusercontent.com/logpai/loghub/master/"
LOG_SETS = ["Android", "Apache", "BGL", "Hadoop", "HDFS", "HealthApp", "HPC", "Linux",
            "Mac", "OpenSSH", "OpenStack", "Proxifier", "Spark", "Thunderbird",
            "Windows", "Zookeeper"]


def download(url):
    """Fetch `url` once into the cache and return its bytes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if not path.exists():
        print("  fetching", url, file=sys.stderr)
        tmp = path.with_suffix(".part")
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-A", "corpus-fetch", "-o", str(tmp), url],
                       check=True)
        tmp.rename(path)
    return path.read_bytes()


def tar_members(data, keep):
    """Concatenate, in name order, the tar members for which keep(name) is true."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and keep(m.name)),
                         key=lambda m: m.name)
        return b"".join(tar.extractfile(m).read() for m in members)


def redis_c():
    return tar_members(download(REDIS),
                       lambda n: "/src/" in n and n.count("/") == 2 and n.endswith(".c"))


def flask_py():
    return tar_members(download(FLASK), lambda n: n.endswith(".py"))


def rg_linux():
    return tar_members(download(RG_LINUX), lambda n: n.endswith("/rg"))


def rg_windows():
    with zipfile.ZipFile(io.BytesIO(download(RG_WIN))) as z:
        name = next(n for n in z.namelist() if n.endswith("/rg.exe"))
        return z.read(name)


def logs():
    return b"".join(download("%s%s/%s_2k.log" % (LOG_BASE, s, s)) for s in LOG_SETS)


def random_bytes():
    return hashlib.shake_256(b"neurozip-benchmark-random-v1").digest(2 * 1024 * 1024)


# name -> (category, description of the source, builder)
CORPORA = {
    "text_war_and_peace.txt": ("English text", "Project Gutenberg #2600 (pglaf.org mirror), plain text",
                               lambda: download("https://gutenberg.pglaf.org/2/6/0/2600/2600-0.txt")),
    "text_moby_dick.txt": ("English text", "Project Gutenberg #2701 (pglaf.org mirror), plain text",
                           lambda: download("https://gutenberg.pglaf.org/2/7/0/2701/2701-0.txt")),
    "code_redis.c": ("Code", "redis 7.2.4, src/*.c concatenated in name order", redis_c),
    "code_flask.py": ("Code", "flask 3.0.0, all *.py concatenated in name order", flask_py),
    "json_citm_catalog.json": ("JSON", "nativejson-benchmark data/citm_catalog.json",
                               lambda: download(JSON_BASE + "citm_catalog.json")),
    "json_twitter.json": ("JSON", "nativejson-benchmark data/twitter.json",
                          lambda: download(JSON_BASE + "twitter.json")),
    "json_canada.json": ("JSON", "nativejson-benchmark data/canada.json",
                         lambda: download(JSON_BASE + "canada.json")),
    "logs_loghub.log": ("Logs", "loghub 2k-line samples of 16 systems, concatenated", logs),
    "exe_rg_linux_x64": ("Executables", "ripgrep 14.1.0 Linux x86-64 musl binary", rg_linux),
    "exe_rg_windows_x64.exe": ("Executables", "ripgrep 14.1.0 Windows x86-64 binary", rg_windows),
    "compressed_redis.tar.gz": ("Compressed", "redis 7.2.4 source tarball (gzip)",
                                lambda: download(REDIS)),
    "compressed_ripgrep.zip": ("Compressed", "ripgrep 14.1.0 Windows release (zip)",
                               lambda: download(RG_WIN)),
    "random_shake256.bin": ("Random", "SHAKE-256 output from a fixed seed", random_bytes),
}


LUA = "https://github.com/lua/lua/archive/refs/tags/v5.4.6.tar.gz"
FD = ("https://github.com/sharkdp/fd/releases/download/v9.0.0/"
      "fd-v9.0.0-x86_64-unknown-linux-musl.tar.gz")
SIMDJSON = "https://raw.githubusercontent.com/simdjson/simdjson-data/master/jsonexamples/"
ELASTIC = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/"

# Development set: the only files that may be used to tune the compressor.
# Different works, projects and log sources from the test set above.
DEV = {
    "text_pride_and_prejudice.txt": ("English text", "Project Gutenberg #1342 (pglaf.org mirror)",
                                     lambda: download("https://gutenberg.pglaf.org/1/3/4/1342/1342-0.txt")),
    "code_lua.c": ("Code", "lua 5.4.6, *.c concatenated in name order",
                   lambda: tar_members(download(LUA), lambda n: n.endswith(".c"))),
    "json_gsoc.json": ("JSON", "simdjson-data jsonexamples/gsoc-2018.json",
                       lambda: download(SIMDJSON + "gsoc-2018.json")),
    "json_github_events.json": ("JSON", "simdjson-data jsonexamples/github_events.json",
                                lambda: download(SIMDJSON + "github_events.json")),
    "logs_nginx.log": ("Logs", "elastic/examples nginx_logs",
                       lambda: download(ELASTIC + "nginx_logs/nginx_logs")),
    "logs_apache.log": ("Logs", "elastic/examples apache_logs",
                        lambda: download(ELASTIC + "apache_logs/apache_logs")),
    "exe_fd_linux_x64": ("Executables", "fd 9.0.0 Linux x86-64 musl binary",
                         lambda: tar_members(download(FD), lambda n: n.endswith("/fd"))),
    "compressed_lua.tar.gz": ("Compressed", "lua 5.4.6 source tarball (gzip)", lambda: download(LUA)),
    "random_dev.bin": ("Random", "SHAKE-256 output from a second fixed seed",
                       lambda: hashlib.shake_256(b"neurozip-dev-random-v1").digest(1024 * 1024)),
}


CURL = "https://github.com/curl/curl/archive/refs/tags/curl-8_5_0.tar.gz"
REQUESTS = "https://github.com/psf/requests/archive/refs/tags/v2.31.0.tar.gz"
SIMDJSON_DATA = "https://raw.githubusercontent.com/simdjson/simdjson-data/master/jsonexamples/"
ZEEK = "https://raw.githubusercontent.com/brimdata/zed-sample-data/main/zeek-default/"
JQ = "https://github.com/jqlang/jq/releases/download/jq-1.7.1/"
BAT_WIN = ("https://github.com/sharkdp/bat/releases/download/v0.24.0/"
           "bat-v0.24.0-x86_64-pc-windows-msvc.zip")

# Second test set. Built after the mixing model and its constants were
# fixed, from works, projects and sources that appear in neither the first
# test set nor the development set. Nothing has been tuned on it.
TEST2 = {
    "text_great_expectations.txt": ("English text", "Project Gutenberg #1400 (pglaf.org mirror)",
                                    lambda: download("https://gutenberg.pglaf.org/1/4/0/1400/1400-0.txt")),
    "text_tale_of_two_cities.txt": ("English text", "Project Gutenberg #98 (pglaf.org mirror)",
                                    lambda: download("https://gutenberg.pglaf.org/9/98/98-0.txt")),
    "code_curl.c": ("Code", "curl 8.5.0, lib/*.c concatenated in name order",
                    lambda: tar_members(download(CURL),
                                        lambda n: "/lib/" in n and n.count("/") == 2 and n.endswith(".c"))),
    "code_requests.py": ("Code", "requests 2.31.0, all *.py concatenated in name order",
                         lambda: tar_members(download(REQUESTS), lambda n: n.endswith(".py"))),
    "json_marine_ik.json": ("JSON", "simdjson-data jsonexamples/marine_ik.json",
                            lambda: download(SIMDJSON_DATA + "marine_ik.json")),
    "json_update_center.json": ("JSON", "simdjson-data jsonexamples/update-center.json",
                                lambda: download(SIMDJSON_DATA + "update-center.json")),
    "json_instruments.json": ("JSON", "simdjson-data jsonexamples/instruments.json",
                              lambda: download(SIMDJSON_DATA + "instruments.json")),
    "logs_zeek_conn.log": ("Logs", "brimdata zed-sample-data zeek-default/conn.log",
                           lambda: gzip.decompress(download(ZEEK + "conn.log.gz"))),
    "logs_zeek_http.log": ("Logs", "brimdata zed-sample-data zeek-default/http.log",
                           lambda: gzip.decompress(download(ZEEK + "http.log.gz"))),
    "exe_jq_linux_x64": ("Executables", "jq 1.7.1 Linux x86-64 binary",
                         lambda: download(JQ + "jq-linux-amd64")),
    "exe_jq_windows_x64.exe": ("Executables", "jq 1.7.1 Windows x86-64 binary",
                               lambda: download(JQ + "jq-windows-amd64.exe")),
    "compressed_jq.tar.gz": ("Compressed", "jq 1.7.1 source tarball (gzip)",
                             lambda: download(JQ + "jq-1.7.1.tar.gz")),
    "compressed_bat.zip": ("Compressed", "bat 0.24.0 Windows release (zip)",
                           lambda: download(BAT_WIN)),
    "random_shake256_b.bin": ("Random", "SHAKE-256 output from a third fixed seed",
                              lambda: hashlib.shake_256(b"neurozip-benchmark-random-v2").digest(
                                  2 * 1024 * 1024)),
}


ZLIB = "https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz"
CLICK = "https://github.com/pallets/click/archive/refs/tags/8.1.7.tar.gz"
BAT_LINUX = ("https://github.com/sharkdp/bat/releases/download/v0.24.0/"
             "bat-v0.24.0-x86_64-unknown-linux-musl.tar.gz")

# Third test set. Built after the model gained bit histories, extra contexts
# and the x86 filter, and after its constants were fixed on the development
# set. None of these files or sources appears in any other set.
TEST3 = {
    "text_sherlock_holmes.txt": ("English text", "Project Gutenberg #1661 (pglaf.org mirror)",
                                 lambda: download("https://gutenberg.pglaf.org/1/6/6/1661/1661-0.txt")),
    "text_frankenstein.txt": ("English text", "Project Gutenberg #84 (pglaf.org mirror)",
                              lambda: download("https://gutenberg.pglaf.org/8/84/84-0.txt")),
    "code_zlib.c": ("Code", "zlib 1.3.1, top-level *.c concatenated in name order",
                    lambda: tar_members(download(ZLIB),
                                        lambda n: n.count("/") == 1 and n.endswith(".c"))),
    "code_click.py": ("Code", "click 8.1.7, all *.py concatenated in name order",
                      lambda: tar_members(download(CLICK), lambda n: n.endswith(".py"))),
    "json_mesh.json": ("JSON", "simdjson-data jsonexamples/mesh.json",
                       lambda: download(SIMDJSON_DATA + "mesh.json")),
    "json_apache_builds.json": ("JSON", "simdjson-data jsonexamples/apache_builds.json",
                                lambda: download(SIMDJSON_DATA + "apache_builds.json")),
    "json_random.json": ("JSON", "simdjson-data jsonexamples/random.json",
                         lambda: download(SIMDJSON_DATA + "random.json")),
    "logs_zeek_dns.log": ("Logs", "brimdata zed-sample-data zeek-default/dns.log",
                          lambda: gzip.decompress(download(ZEEK + "dns.log.gz"))),
    "logs_zeek_ssl.log": ("Logs", "brimdata zed-sample-data zeek-default/ssl.log",
                          lambda: gzip.decompress(download(ZEEK + "ssl.log.gz"))),
    "exe_busybox_linux_x64": ("Executables", "busybox 1.35.0 Linux x86-64 musl binary",
                              lambda: download("https://busybox.net/downloads/binaries/"
                                               "1.35.0-x86_64-linux-musl/busybox")),
    "exe_bat_linux_x64": ("Executables", "bat 0.24.0 Linux x86-64 musl binary",
                          lambda: tar_members(download(BAT_LINUX), lambda n: n.endswith("/bat"))),
    "compressed_zlib.tar.gz": ("Compressed", "zlib 1.3.1 source tarball (gzip)",
                               lambda: download(ZLIB)),
    "compressed_click.tar.gz": ("Compressed", "click 8.1.7 source tarball (gzip)",
                                lambda: download(CLICK)),
    "random_shake256_c.bin": ("Random", "SHAKE-256 output from a fourth fixed seed",
                              lambda: hashlib.shake_256(b"neurozip-benchmark-random-v3").digest(
                                  2 * 1024 * 1024)),
}


def main():
    build_set(CORPORA, OUT, LOCK)
    build_set(TEST3, HERE / "corpora3", HERE / "corpora3.lock.json")
    build_set(TEST2, HERE / "corpora2", HERE / "corpora2.lock.json")
    build_set(DEV, HERE / "dev", HERE / "dev.lock.json")


def build_set(corpora, out_dir, lock_path):
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, (category, source, build) in corpora.items():
        full = build()
        data = full[:CAP]
        (out_dir / name).write_bytes(data)
        built[name] = {
            "category": category,
            "source": source,
            "bytes": len(data),
            "truncated_from": len(full) if len(full) > CAP else None,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        print("%-28s %-13s %9d bytes" % (name, category, len(data)), file=sys.stderr)

    if lock_path.exists():
        locked = json.loads(lock_path.read_text())
        bad = [n for n in built if locked.get(n, {}).get("sha256") != built[n]["sha256"]]
        if bad or set(locked) != set(built):
            sys.exit("files differ from %s: %s" % (lock_path.name, bad or "file list changed"))
        print("all files match", lock_path.name, file=sys.stderr)
    else:
        lock_path.write_text(json.dumps(built, indent=2) + "\n")
        print("wrote", lock_path, file=sys.stderr)


if __name__ == "__main__":
    main()
