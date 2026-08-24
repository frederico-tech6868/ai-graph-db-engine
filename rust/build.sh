#!/bin/bash
# Build the graphdb_rs Rust extension and install it into the current Python env.
set -e

echo "Installing maturin..."
pip install maturin -q

echo "Building Rust extension (release)..."
cd "$(dirname "$0")"
maturin develop --release

echo "Build complete. Backend: Rust"
