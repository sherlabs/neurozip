#!/bin/sh
# Bits per byte on the compressible development files, and their unweighted
# mean. A tuning aid: the benchmark files are never used here.
cd "$(dirname "$0")/.." || exit 1
for f in bench/dev/text* bench/dev/code* bench/dev/json* bench/dev/logs* bench/dev/exe*; do
  ./target/release/neurozip ablate "$f" --model mix "$@" &
done | sed 's/.*"file":"\([^"]*\)".*"bits_per_byte":\([0-9.]*\).*/\2 \1/' | sort -k2 |
  awk '{t+=$1; n++; printf "%s=%.3f ", substr($2,1,9), $1} END {printf "\nMEAN %.4f\n", t/n}'
wait
