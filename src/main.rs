//! `neurozip` command line: compress, decompress, bench.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use neurozip::model::mix;
use neurozip::{
    Ablation, CtxParams, Limits, MixParams, ModelSpec, Options, Stats, ablation_size, compress,
    compress_stream, decompress, decompress_stream, sha256,
};

/// Names accepted by `--mix-off`, with the header flag each one sets.
const MIX_SWITCHES: [(u8, &str); 5] = [
    (mix::FLAG_NO_X86, "x86"),
    (mix::FLAG_NO_MATCH, "match"),
    (mix::FLAG_NO_APM, "apm"),
    (mix::FLAG_NO_WORD, "word"),
    (mix::FLAG_FROZEN_MIXER, "mixer"),
];

const USAGE: &str = "\
neurozip - experimental lossless compressor (online-learned model + range coder)

USAGE:
    neurozip compress   <input> [-o <output>] [options]
    neurozip decompress <input> [-o <output>] [options]
    neurozip bench      <file>... [options]
    neurozip ablate     <file>... [options]

Use '-' as <input> for stdin and '-o -' for stdout.

COMPRESS OPTIONS:
    -o, --output <path>     Output file (default: <input>.nz)
    -f, --force             Overwrite an existing output file
        --model <name>      mix (default), ctx or stored
        --hash-bits <n>     log2 table size per context
                            (mix: 10-24, default 18; ctx: 8-24, default 20)
        --match-bits <n>    mix: log2 match window in bytes, 12-28 (default 22)
        --mix-off <parts>   mix: switch parts off, comma separated:
                            match, apm, word, x86, mixer (mixer = no learning)
        --order <n>         ctx: highest context order, 0-8 (default 4)
        --block-size <n>    Raw bytes per block; suffix K or M allowed
                            (default 256K)
        --independent-blocks
                            Reset the model at every block

DECOMPRESS OPTIONS:
    -o, --output <path>     Output file (default: <input> without .nz)
    -f, --force             Overwrite an existing output file
        --max-memory <mib>  Refuse archives needing more model memory
                            (default 1024)

BENCH OPTIONS:
    Same model options as compress, plus:
        --orders <a,b,..>   ctx: run once per listed order
        --json              Machine-readable output, one object per line

ABLATE OPTIONS:
    Same model options as compress, plus:
        --orders <a,b,..>   ctx: run once per listed order
        --no-fallback       Range-code every block, never store one raw
        --freeze-after <p>  ctx: stop learning after p percent of the input

Ablate reports, as JSON lines, the exact archive size an encoder variant
would produce. Only the default variant is a real, decodable archive.

Bench compresses and decompresses in memory, verifies the round trip
byte for byte and by SHA-256, and reports sizes that include all archive
metadata. Timings are single runs on this machine.
";

/// Model used when `--model` is not given.
const DEFAULT_MODEL: &str = "mix";

type CliResult<T> = Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("neurozip: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> CliResult<()> {
    let Some(cmd) = args.first() else {
        return Err(format!("missing command\n\n{USAGE}"));
    };
    match cmd.as_str() {
        "compress" | "c" => cmd_compress(&parse_args(&args[1..])?),
        "decompress" | "d" => cmd_decompress(&parse_args(&args[1..])?),
        "bench" | "b" => cmd_bench(&parse_args(&args[1..])?),
        "ablate" => cmd_ablate(&parse_args(&args[1..])?),
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        "-V" | "--version" => {
            println!("neurozip {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
}

#[derive(Default)]
struct Args {
    inputs: Vec<String>,
    output: Option<String>,
    force: bool,
    model: Option<String>,
    order: Option<u8>,
    orders: Option<Vec<u8>>,
    hash_bits: Option<u8>,
    block_size: Option<u32>,
    independent_blocks: bool,
    max_memory_mib: Option<u64>,
    json: bool,
    no_fallback: bool,
    match_bits: Option<u8>,
    mix_off: Option<String>,
    freeze_after_pct: Option<f64>,
}

fn parse_args(raw: &[String]) -> CliResult<Args> {
    let mut a = Args::default();
    let mut it = raw.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> CliResult<&String> {
            it.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "-o" | "--output" => a.output = Some(value(arg)?.clone()),
            "-f" | "--force" => a.force = true,
            "--model" => a.model = Some(value(arg)?.clone()),
            "--order" => a.order = Some(parse_num(arg, value(arg)?)?),
            "--orders" => {
                let list = value(arg)?
                    .split(',')
                    .map(|s| parse_num::<u8>(arg, s))
                    .collect::<CliResult<Vec<u8>>>()?;
                a.orders = Some(list);
            }
            "--hash-bits" => a.hash_bits = Some(parse_num(arg, value(arg)?)?),
            "--block-size" => a.block_size = Some(parse_size(value(arg)?)?),
            "--independent-blocks" => a.independent_blocks = true,
            "--max-memory" => a.max_memory_mib = Some(parse_num(arg, value(arg)?)?),
            "--json" => a.json = true,
            "--match-bits" => a.match_bits = Some(parse_num(arg, value(arg)?)?),
            "--mix-off" => a.mix_off = Some(value(arg)?.clone()),
            "--no-fallback" => a.no_fallback = true,
            "--freeze-after" => {
                let p: f64 = parse_num(arg, value(arg)?)?;
                if !(0.0..=100.0).contains(&p) {
                    return Err("--freeze-after takes a percentage from 0 to 100".into());
                }
                a.freeze_after_pct = Some(p);
            }
            "-" => a.inputs.push(arg.clone()),
            s if s.starts_with('-') => return Err(format!("unknown option '{s}'")),
            _ => a.inputs.push(arg.clone()),
        }
    }
    Ok(a)
}

fn parse_num<T: std::str::FromStr>(name: &str, s: &str) -> CliResult<T> {
    s.trim()
        .parse()
        .map_err(|_| format!("invalid value '{s}' for {name}"))
}

fn parse_size(s: &str) -> CliResult<u32> {
    let s = s.trim();
    let (digits, mult) = match s.chars().last() {
        Some('k' | 'K') => (&s[..s.len() - 1], 1u64 << 10),
        Some('m' | 'M') => (&s[..s.len() - 1], 1u64 << 20),
        _ => (s, 1),
    };
    let n: u64 = parse_num("--block-size", digits)?;
    u32::try_from(n.saturating_mul(mult)).map_err(|_| format!("block size '{s}' is too large"))
}

/// Build encoder options from the command line; `order` overrides `--order`.
fn options(a: &Args, order: Option<u8>) -> CliResult<Options> {
    let defaults = CtxParams::default();
    let mix_defaults = MixParams::default();
    let mut mix_flags = 0u8;
    for name in a.mix_off.iter().flat_map(|s| s.split(',')) {
        match MIX_SWITCHES.iter().find(|(_, label)| *label == name) {
            Some((flag, _)) => mix_flags |= flag,
            None => {
                return Err(format!(
                    "unknown --mix-off part '{name}' (expected match, apm, word, x86 or mixer)"
                ));
            }
        }
    }
    let model = match a.model.as_deref().unwrap_or(DEFAULT_MODEL) {
        "mix" => ModelSpec::Mix(MixParams {
            hash_bits: a.hash_bits.unwrap_or(mix_defaults.hash_bits),
            match_bits: a.match_bits.unwrap_or(mix_defaults.match_bits),
            flags: mix_flags,
        }),
        "ctx" => ModelSpec::Ctx(CtxParams {
            max_order: order.or(a.order).unwrap_or(defaults.max_order),
            hash_bits: a.hash_bits.unwrap_or(defaults.hash_bits),
        }),
        "stored" => ModelSpec::None,
        other => {
            return Err(format!(
                "unknown model '{other}' (expected mix, ctx or stored)"
            ));
        }
    };
    let opts = Options {
        model,
        block_size: a.block_size.unwrap_or(Options::default().block_size),
        independent_blocks: a.independent_blocks,
    };
    opts.validate().map_err(|e| e.to_string())?;
    Ok(opts)
}

fn single_input(a: &Args) -> CliResult<&str> {
    match a.inputs.as_slice() {
        [one] => Ok(one),
        [] => Err("missing input file".into()),
        _ => Err("expected exactly one input file".into()),
    }
}

/// Where output goes. A file is written under a temporary name and renamed
/// into place only on success, so a failed run never leaves a partial or
/// corrupt result behind.
enum Sink {
    Stdout,
    File { tmp: PathBuf, dest: PathBuf },
}

impl Sink {
    fn open(dest: &str, force: bool) -> CliResult<(Sink, Box<dyn Write>)> {
        if dest == "-" {
            return Ok((Sink::Stdout, Box::new(BufWriter::new(io::stdout().lock()))));
        }
        let dest = PathBuf::from(dest);
        if dest.exists() && !force {
            return Err(format!(
                "{} already exists (use -f to overwrite)",
                dest.display()
            ));
        }
        let mut name = dest.file_name().unwrap_or_default().to_os_string();
        name.push(format!(".tmp{}", std::process::id()));
        let tmp = dest.with_file_name(name);
        let file =
            File::create(&tmp).map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        Ok((Sink::File { tmp, dest }, Box::new(BufWriter::new(file))))
    }

    fn commit(self) -> CliResult<()> {
        if let Sink::File { tmp, dest } = self {
            fs::rename(&tmp, &dest).map_err(|e| {
                let _ = fs::remove_file(&tmp);
                format!("cannot write {}: {e}", dest.display())
            })?;
        }
        Ok(())
    }

    fn discard(self) {
        if let Sink::File { tmp, .. } = self {
            let _ = fs::remove_file(tmp);
        }
    }
}

fn open_input(path: &str) -> CliResult<File> {
    File::open(path).map_err(|e| format!("cannot open {path}: {e}"))
}

fn cmd_compress(a: &Args) -> CliResult<()> {
    let input = single_input(a)?;
    let opts = options(a, None)?;
    let output = match (&a.output, input) {
        (Some(o), _) => o.clone(),
        (None, "-") => "-".to_string(),
        (None, path) => format!("{path}.nz"),
    };

    // The format records the length up front, so stdin is buffered.
    let (reader, len): (Box<dyn Read>, u64) = if input == "-" {
        let mut buf = Vec::new();
        io::stdin()
            .lock()
            .read_to_end(&mut buf)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        let len = buf.len() as u64;
        (Box::new(io::Cursor::new(buf)), len)
    } else {
        let file = open_input(input)?;
        let meta = file
            .metadata()
            .map_err(|e| format!("cannot stat {input}: {e}"))?;
        if !meta.is_file() {
            return Err(format!("{input} is not a regular file"));
        }
        (Box::new(BufReader::new(file)), meta.len())
    };

    let (sink, writer) = Sink::open(&output, a.force)?;
    let start = Instant::now();
    match compress_stream(reader, writer, len, &opts) {
        Ok(stats) => {
            sink.commit()?;
            report("compressed", &stats, start);
            Ok(())
        }
        Err(e) => {
            sink.discard();
            Err(format!("{input}: {e}"))
        }
    }
}

fn cmd_decompress(a: &Args) -> CliResult<()> {
    let input = single_input(a)?;
    let output = match (&a.output, input) {
        (Some(o), _) => o.clone(),
        (None, "-") => "-".to_string(),
        (None, path) => match path.strip_suffix(".nz") {
            Some(stem) if !stem.is_empty() => stem.to_string(),
            _ => {
                return Err(format!(
                    "{path} does not end in .nz; give an output with -o"
                ));
            }
        },
    };
    let limits = Limits {
        max_model_memory: a
            .max_memory_mib
            .map(|m| m.saturating_mul(1 << 20))
            .unwrap_or(Limits::default().max_model_memory),
        ..Limits::default()
    };
    let reader: Box<dyn Read> = if input == "-" {
        Box::new(BufReader::new(io::stdin().lock()))
    } else {
        Box::new(BufReader::new(open_input(input)?))
    };

    let (sink, writer) = Sink::open(&output, a.force)?;
    let start = Instant::now();
    match decompress_stream(reader, writer, &limits) {
        Ok(stats) => {
            sink.commit()?;
            report("decompressed", &stats, start);
            Ok(())
        }
        Err(e) => {
            sink.discard();
            Err(format!("{input}: {e}"))
        }
    }
}

fn report(verb: &str, s: &Stats, start: Instant) {
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "{verb}: {} -> {} bytes archive ({:.3} bits/byte, {} coded + {} stored blocks) in {:.2}s ({:.2} MB/s)",
        s.original_len,
        s.archive_len,
        bits_per_byte(s.archive_len, s.original_len),
        s.blocks_coded,
        s.blocks_stored,
        secs,
        mb_per_s(s.original_len, secs),
    );
}

fn bits_per_byte(archive: u64, original: u64) -> f64 {
    if original == 0 {
        0.0
    } else {
        archive as f64 * 8.0 / original as f64
    }
}

fn mb_per_s(bytes: u64, secs: f64) -> f64 {
    if secs > 0.0 {
        bytes as f64 / 1e6 / secs
    } else {
        0.0
    }
}

fn cmd_bench(a: &Args) -> CliResult<()> {
    if a.inputs.is_empty() {
        return Err("bench needs at least one file".into());
    }
    let orders: Vec<Option<u8>> = match &a.orders {
        Some(list) => list.iter().map(|&o| Some(o)).collect(),
        None => vec![None],
    };
    if !a.json {
        println!(
            "{:<28} {:>14} {:>12} {:>12} {:>7} {:>9} {:>9} {:>8} {:>9}",
            "file",
            "config",
            "original",
            "archive",
            "bpb",
            "enc MB/s",
            "dec MB/s",
            "model MB",
            "blocks c/s"
        );
    }
    for path in &a.inputs {
        let data = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        for &order in &orders {
            let opts = options(a, order)?;
            bench_one(path, &data, &opts, a.json)?;
        }
    }
    Ok(())
}

fn cmd_ablate(a: &Args) -> CliResult<()> {
    if a.inputs.is_empty() {
        return Err("ablate needs at least one file".into());
    }
    let orders: Vec<Option<u8>> = match &a.orders {
        Some(list) => list.iter().map(|&o| Some(o)).collect(),
        None => vec![None],
    };
    for path in &a.inputs {
        let data = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let ablation = Ablation {
            fallback: !a.no_fallback,
            freeze_after: a
                .freeze_after_pct
                .map(|p| (data.len() as f64 * p / 100.0) as u64),
        };
        for &order in &orders {
            let opts = options(a, order)?;
            let size =
                ablation_size(&data, &opts, &ablation).map_err(|e| format!("{path}: {e}"))?;
            println!(
                "{{\"file\":{:?},\"config\":{:?},\"fallback\":{},\"freeze_after_bytes\":{},\
                 \"original_bytes\":{},\"archive_bytes\":{size},\"bits_per_byte\":{:.6}}}",
                file_name(path),
                config_name(&opts),
                ablation.fallback,
                ablation
                    .freeze_after
                    .map_or("null".to_string(), |n| n.to_string()),
                data.len(),
                bits_per_byte(size, data.len() as u64),
            );
        }
    }
    Ok(())
}

fn config_name(opts: &Options) -> String {
    match opts.model {
        ModelSpec::None => "stored".to_string(),
        ModelSpec::Ctx(p) => format!("ctx-o{}-h{}", p.max_order, p.hash_bits),
        ModelSpec::Mix(p) => {
            let mut name = format!("mix-h{}-m{}", p.hash_bits, p.match_bits);
            for (flag, label) in MIX_SWITCHES {
                if p.flags & flag != 0 {
                    name.push_str("-no");
                    name.push_str(label);
                }
            }
            name
        }
    }
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn bench_one(path: &str, data: &[u8], opts: &Options, json: bool) -> CliResult<()> {
    let limits = Limits {
        max_model_memory: u64::MAX,
        ..Limits::default()
    };
    let t = Instant::now();
    let archive = compress(data, opts).map_err(|e| format!("{path}: compress failed: {e}"))?;
    let enc_s = t.elapsed().as_secs_f64();

    let mut restored = Vec::with_capacity(data.len());
    let t = Instant::now();
    let stats = decompress_stream(&archive[..], &mut restored, &limits)
        .map_err(|e| format!("{path}: decompress failed: {e}"))?;
    let dec_s = t.elapsed().as_secs_f64();

    // Lossless reconstruction is checked twice: bytes and digest.
    if restored != data || sha256(&restored) != sha256(data) {
        return Err(format!("{path}: ROUND TRIP MISMATCH"));
    }
    // And the in-memory convenience path must agree with the stream path.
    debug_assert_eq!(decompress(&archive, &limits).ok().as_deref(), Some(data));

    let config = config_name(opts);
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let orig = data.len() as u64;
    let arch = archive.len() as u64;
    let bpb = bits_per_byte(arch, orig);
    let model_mb = opts.model.memory_bytes() as f64 / (1 << 20) as f64;
    if json {
        println!(
            "{{\"file\":{:?},\"config\":{:?},\"original_bytes\":{orig},\"archive_bytes\":{arch},\
             \"bits_per_byte\":{bpb:.6},\"encode_mb_s\":{:.4},\"decode_mb_s\":{:.4},\
             \"model_bytes\":{},\"blocks_coded\":{},\"blocks_stored\":{},\"verified\":true}}",
            name,
            config,
            mb_per_s(orig, enc_s),
            mb_per_s(orig, dec_s),
            opts.model.memory_bytes(),
            stats.blocks_coded,
            stats.blocks_stored,
        );
    } else {
        println!(
            "{:<28} {:>14} {:>12} {:>12} {:>7.3} {:>9.2} {:>9.2} {:>8.1} {:>9}",
            truncate(&name, 28),
            config,
            orig,
            arch,
            bpb,
            mb_per_s(orig, enc_s),
            mb_per_s(orig, dec_s),
            model_mb,
            format!("{}/{}", stats.blocks_coded, stats.blocks_stored),
        );
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let tail: String = s
            .chars()
            .rev()
            .take(max - 1)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("~{tail}")
    }
}
