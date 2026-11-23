//! End-to-end tests of the `neurozip` binary.

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

use common::{random_bytes, text_bytes};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_neurozip"))
}

/// A fresh scratch directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("neurozip-cli-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(cmd: &mut Command) -> Output {
    cmd.output().expect("failed to run neurozip")
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

const SMALL: [&str; 2] = ["--hash-bits", "12"];

#[test]
fn compress_then_decompress_files() {
    let dir = Scratch::new("files");
    let input = dir.path("data.txt");
    let mut data = text_bytes(300_000, 1);
    data.extend(random_bytes(50_000, 2));
    fs::write(&input, &data).unwrap();

    ok(&run(bin().arg("compress").arg(&input).args(SMALL)));
    let archive = dir.path("data.txt.nz");
    let archive_len = fs::metadata(&archive).unwrap().len();
    assert!(archive_len < data.len() as u64);

    let restored = dir.path("restored.txt");
    ok(&run(bin()
        .arg("decompress")
        .arg(&archive)
        .arg("-o")
        .arg(&restored)));
    assert!(fs::read(&restored).unwrap() == data);

    // Default output name strips ".nz"; it exists, so this must refuse...
    let out = run(bin().arg("decompress").arg(&archive));
    assert!(!out.status.success());
    assert!(
        fs::read(&input).unwrap() == data,
        "refusal must not touch the file"
    );
    // ...unless forced.
    ok(&run(bin().arg("decompress").arg(&archive).arg("-f")));
    assert!(fs::read(&input).unwrap() == data);
}

#[test]
fn empty_file() {
    let dir = Scratch::new("empty");
    let input = dir.path("empty");
    fs::write(&input, b"").unwrap();
    ok(&run(bin().arg("compress").arg(&input).args(SMALL)));
    let restored = dir.path("out");
    ok(&run(bin()
        .arg("decompress")
        .arg(dir.path("empty.nz"))
        .arg("-o")
        .arg(&restored)));
    assert_eq!(fs::read(&restored).unwrap(), b"");
}

#[test]
fn stdin_to_stdout() {
    use std::io::Write;
    let data = text_bytes(50_000, 5);

    let pipe = |args: &[&str], input: &[u8]| -> Vec<u8> {
        let mut child = bin()
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let input = input.to_vec();
        let feeder = std::thread::spawn(move || stdin.write_all(&input));
        let out = child.wait_with_output().unwrap();
        feeder.join().unwrap().unwrap();
        ok(&out);
        out.stdout
    };

    let archive = pipe(&["compress", "-", "--hash-bits", "12"], &data);
    assert!(archive.starts_with(b"NZIP"));
    let restored = pipe(&["decompress", "-"], &archive);
    assert!(restored == data);
}

#[test]
fn corrupt_archive_fails_and_leaves_no_output() {
    let dir = Scratch::new("corrupt");
    let input = dir.path("data");
    fs::write(&input, text_bytes(100_000, 3)).unwrap();
    ok(&run(bin().arg("compress").arg(&input).args(SMALL)));

    let archive = dir.path("data.nz");
    let mut bytes = fs::read(&archive).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x10;
    fs::write(&archive, &bytes).unwrap();

    let restored = dir.path("restored");
    let out = run(bin()
        .arg("decompress")
        .arg(&archive)
        .arg("-o")
        .arg(&restored));
    assert!(!out.status.success());
    assert!(
        !restored.exists(),
        "a failed decompress must not leave output"
    );
    let leftovers: Vec<_> = fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.contains(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temporary files left: {leftovers:?}");
}

#[test]
fn memory_limit_flag() {
    let dir = Scratch::new("mem");
    let input = dir.path("data");
    fs::write(&input, text_bytes(10_000, 3)).unwrap();
    // order 4, hash-bits 16 needs 3 x 2 MiB plus the order-1 table.
    ok(&run(bin().arg("compress").arg(&input).args([
        "--model",
        "ctx",
        "--hash-bits",
        "16",
    ])));
    let archive = dir.path("data.nz");
    let restored = dir.path("restored");
    let out = run(bin()
        .arg("decompress")
        .arg(&archive)
        .arg("-o")
        .arg(&restored)
        .args(["--max-memory", "4"]));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("model memory"));
    ok(&run(bin()
        .arg("decompress")
        .arg(&archive)
        .arg("-o")
        .arg(&restored)
        .args(["--max-memory", "8"])));
}

#[test]
fn compress_options_are_honoured() {
    let dir = Scratch::new("opts");
    let input = dir.path("data");
    let data = text_bytes(60_000, 8);
    fs::write(&input, &data).unwrap();
    let mut sizes = Vec::new();
    for (i, args) in [
        vec!["--model", "stored"],
        vec!["--model", "ctx", "--order", "0"],
        vec![
            "--model",
            "ctx",
            "--order",
            "3",
            "--hash-bits",
            "12",
            "--block-size",
            "4K",
        ],
        vec![
            "--model",
            "ctx",
            "--order",
            "3",
            "--hash-bits",
            "12",
            "--block-size",
            "4K",
            "--independent-blocks",
        ],
        vec![
            "--hash-bits",
            "12",
            "--match-bits",
            "14",
            "--mix-off",
            "match,apm",
        ],
        vec!["--model", "mix", "--hash-bits", "12", "--match-bits", "14"],
    ]
    .iter()
    .enumerate()
    {
        let archive = dir.path(&format!("a{i}.nz"));
        ok(&run(bin()
            .arg("compress")
            .arg(&input)
            .arg("-o")
            .arg(&archive)
            .args(args)));
        let restored = dir.path(&format!("r{i}"));
        ok(&run(bin()
            .arg("decompress")
            .arg(&archive)
            .arg("-o")
            .arg(&restored)));
        assert!(fs::read(&restored).unwrap() == data);
        sizes.push(fs::metadata(&archive).unwrap().len());
    }
    assert!(sizes[0] > data.len() as u64, "stored adds metadata");
    assert!(sizes[1] < sizes[0]);
    assert!(sizes[2] < sizes[1]);
    assert!(sizes[3] > sizes[2]);
    // The full mixing model is smaller than the same model with parts off.
    assert!(sizes[5] < sizes[4]);
}

#[test]
fn bench_verifies_and_reports() {
    let dir = Scratch::new("bench");
    let a = dir.path("text");
    let b = dir.path("random");
    fs::write(&a, text_bytes(80_000, 1)).unwrap();
    fs::write(&b, random_bytes(20_000, 2)).unwrap();

    let out = run(bin().arg("bench").arg(&a).arg(&b).args([
        "--model",
        "ctx",
        "--orders",
        "0,2",
        "--hash-bits",
        "12",
        "--json",
    ]));
    ok(&out);
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 4);
    for line in &lines {
        assert!(line.starts_with('{') && line.ends_with('}'));
        assert!(line.contains("\"verified\":true"));
    }
    assert!(lines[0].contains("\"config\":\"ctx-o0-h12\""));
    assert!(lines[3].contains("\"original_bytes\":20000"));
    // Random data: every block stored, archive = data + metadata.
    assert!(lines[3].contains("\"blocks_coded\":0"));

    let out = run(bin().arg("bench").arg(&a).args(SMALL));
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).contains("bpb"));
}

#[test]
fn usage_errors() {
    assert!(!run(&mut bin()).status.success());
    assert!(!run(bin().arg("frobnicate")).status.success());
    assert!(!run(bin().arg("compress")).status.success());
    assert!(
        !run(bin().args(["compress", "x", "--model", "ctx", "--order", "99"]))
            .status
            .success()
    );
    assert!(
        !run(bin().args(["compress", "/nonexistent/definitely/missing"]))
            .status
            .success()
    );
    assert!(
        !run(bin().args(["decompress", "no-suffix"]))
            .status
            .success()
    );
    assert!(
        !run(bin().args(["compress", "x", "--mix-off", "everything"]))
            .status
            .success()
    );
    assert!(
        !run(bin().args(["compress", "x", "--model", "nope"]))
            .status
            .success()
    );
    ok(&run(bin().arg("--help")));
    ok(&run(bin().arg("--version")));
}
