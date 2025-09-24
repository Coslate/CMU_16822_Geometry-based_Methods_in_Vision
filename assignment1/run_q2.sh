#!/usr/bin/env bash
set -euo pipefail

# Annotations
ANN_Q1="data/annotation/q1_annotation.npy"
ANN_Q2="data/annotation/q2_annotation.npy"

# Output root for everything in this run
OUTBASE="out_q2"

# Images you want to process (names correspond to keys inside the .npy files)
for k in chess1 book1 checker1 tiles5; do
  # find the image file in data/q1/ with a common extension
  f=""
  for ext in jpg jpeg png JPG JPEG PNG; do
    test -f "data/q1/${k}.${ext}" && f="data/q1/${k}.${ext}" && break
  done
  if [[ -z "$f" ]]; then
    echo "data/q1/${k}.[jpg|jpeg|png] not found — skipping"
    continue
  fi

  echo "> Processing $k  (file: $f)"

  # Per-image output dirs (all under ./out_q2/)
  OUTROOT="${OUTBASE}/${k}"
  AFF_DIR="${OUTROOT}/affine"
  MET_DIR="${OUTROOT}/metric"
  mkdir -p "$AFF_DIR" "$MET_DIR"

  # ---------- Q1: affine rectification (to get H_aff.npy) ----------
  python affine_rectify.py \
    --img "$f" \
    --ann "$ANN_Q1" \
    --key "$k" \
    --outdir "$AFF_DIR"

  if [[ ! -f "${AFF_DIR}/H_aff.npy" ]]; then
    echo "ERROR: ${AFF_DIR}/H_aff.npy not found; affine stage failed for '$k'." >&2
    continue
  fi

  # ---------- Q2: metric rectification (uses H_aff from Q1) ----------
  python metric_rectify_from_affine.py \
    --image "$f" \
    --ann "$ANN_Q2" \
    --key "$k" \
    --H_aff "${AFF_DIR}/H_aff.npy" \
    --out "$MET_DIR"

  echo "> Done: outputs in ${OUTROOT}/affine and ${OUTROOT}/metric"
  echo ""
done