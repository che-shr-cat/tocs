#!/usr/bin/env bash
# Build arxiv submission package from paper/ directory.
#
# Usage:
#   ./scripts/build_arxiv.sh
#
# Produces: paper/tocs-arxiv.tar.gz ready for upload to arxiv.org
#
# Steps:
#   1. Compile LaTeX locally (pdflatex + bibtex) to generate .bbl
#   2. Verify compilation succeeded
#   3. Package only the files arxiv needs into a .tar.gz
#   4. List contents for review

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")/../paper" && pwd)"
BUILD_DIR="$(mktemp -d)"
OUTPUT="$PAPER_DIR/tocs-arxiv.tar.gz"

echo "=== Building arxiv package ==="
echo "Paper dir: $PAPER_DIR"
echo "Build dir: $BUILD_DIR"
echo ""

# ── 1. Compile to generate .bbl ─────────────────────────────────
echo "--- Compiling LaTeX ---"
cd "$PAPER_DIR"

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Install texlive:"
    echo "  sudo apt install texlive-full  # Ubuntu/WSL"
    echo "  brew install --cask mactex     # macOS"
    echo ""
    echo "Or if you already have a .bbl file, place it in paper/ and re-run."
    exit 1
fi

pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

# Check that .bbl was generated
if [ ! -f main.bbl ]; then
    echo "WARNING: main.bbl not generated. References may not render on arxiv."
    echo "         Continuing anyway — you can add .bbl manually later."
fi

# Check that PDF was generated (sanity check)
if [ -f main.pdf ]; then
    PAGES=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "?")
    echo "Compiled successfully: main.pdf ($PAGES pages)"
else
    echo "WARNING: PDF not generated. Check for LaTeX errors:"
    echo "  cd paper/ && pdflatex main.tex"
fi
echo ""

# ── 2. Assemble package ─────────────────────────────────────────
echo "--- Assembling package ---"

# Copy source files (no auxiliary files, no PDF)
cp main.tex "$BUILD_DIR/"
cp references.bib "$BUILD_DIR/"
cp neurips_2025.sty "$BUILD_DIR/"

# Copy .bbl if it exists
if [ -f main.bbl ]; then
    cp main.bbl "$BUILD_DIR/"
    echo "  + main.bbl (compiled references)"
fi

# Copy figures
mkdir -p "$BUILD_DIR/figures"
for fig in figures/*.png figures/*.pdf figures/*.eps figures/*.jpg 2>/dev/null; do
    if [ -f "$fig" ]; then
        cp "$fig" "$BUILD_DIR/$fig"
        echo "  + $fig"
    fi
done

echo "  + main.tex"
echo "  + references.bib"
echo "  + neurips_2025.sty"
echo ""

# ── 3. Create tarball ────────────────────────────────────────────
echo "--- Creating archive ---"
cd "$BUILD_DIR"
tar czf "$OUTPUT" ./*

echo "Created: $OUTPUT"
echo ""

# ── 4. Verify contents ──────────────────────────────────────────
echo "--- Archive contents ---"
tar tzf "$OUTPUT"
echo ""

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "Size: $SIZE"
echo ""

# ── 5. Cleanup ───────────────────────────────────────────────────
rm -rf "$BUILD_DIR"

# ── 6. Pre-flight checks ────────────────────────────────────────
echo "=== Pre-flight checks ==="

# Check for common issues
WARNINGS=0

# Check figure references match files
for fig in $(grep -oP 'includegraphics.*?\{.*?\}' main.tex | grep -oP '\{[^}]+\}' | tr -d '{}'); do
    if [ ! -f "$fig" ]; then
        echo "WARNING: Referenced figure not found: $fig"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Check for [tbd] placeholders
if grep -q '\[tbd\]' main.tex; then
    echo "WARNING: Found [tbd] placeholder in main.tex"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for "anonymous" in author
if grep -qi 'anonymous' main.tex | head -5; then
    echo "WARNING: Found 'anonymous' in main.tex — update author info?"
    WARNINGS=$((WARNINGS + 1))
fi

if [ "$WARNINGS" -eq 0 ]; then
    echo "All checks passed!"
else
    echo "$WARNINGS warning(s) found — review before submitting."
fi

echo ""
echo "=== Done ==="
echo "Upload $OUTPUT to https://arxiv.org/submit"
echo "Recommended categories: cs.SE (primary), cs.AI (cross-list)"
