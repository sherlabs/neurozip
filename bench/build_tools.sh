#!/bin/sh
# Fetch and build lpaq1 (Matt Mahoney, GPL) for the benchmark. It is run as
# a separate program and is not part of NeuroZip. PPMd comes from 7-Zip
# (`7zz`) and zpaq from its own package; install those with a package
# manager.
set -e
cd "$(dirname "$0")"
mkdir -p tools
cd tools
if [ ! -f lpaq1.cpp ]; then
  curl -fsSL -o lpaq1.zip https://mattmahoney.net/dc/lpaq1.zip
  unzip -o -q lpaq1.zip lpaq1.cpp
fi
c++ -O2 -w -o lpaq1 lpaq1.cpp
shasum -a 256 lpaq1.cpp
echo "built $(pwd)/lpaq1"
