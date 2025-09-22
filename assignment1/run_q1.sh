# from your assignment root
for k in chess1 book1 checker1; do
  # try common extensions; pick the first that exists
  f=""
  for ext in jpg jpeg png JPG JPEG PNG; do
    test -f "data/q1/${k}.${ext}" && f="data/q1/${k}.${ext}" && break
  done
  if [ -z "$f" ]; then
    echo "data/q1/${k}.[jpg|jpeg|png] not found — skipping"
    continue
  fi

  echo "> Processing $k -> $f"
  python affine_rectify.py \
    --img "$f" \
    --ann data/annotation/q1_annotation.npy \
    --key "$k" \
    --outdir "out_q1_${k}"

  # save 'test lines only' views (red hold-out lines)
done