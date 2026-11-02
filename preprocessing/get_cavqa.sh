#!/bin/bash
# Download CA-VQA (val + 4 train tasks) one task at a time and convert. Needs ~7.1 TB for raw + ~0.8 TB for records.
# Usage: RAW=/projects/x/cavqa/raw OUT=/projects/x/cavqa/out bash preprocessing/get_cavqa.sh
# DELETE_RAW=1 removes each task's raw files once it is converted and checked. Keep $RAW/cavqa_val: val records point to its images.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${RAW:?set RAW}" "${OUT:?set OUT}"
NP=${NP:-16}
mkdir -p "$RAW" "$OUT" "$OUT/logs"
[ -d "$RAW/ml-cubifyanything" ] || git clone --depth 1 https://github.com/apple/ml-cubifyanything.git "$RAW/ml-cubifyanything"
LINKS=$RAW/ml-cubifyanything/data/cavqa

if [ ! -f "$OUT/.val_done" ]; then
  wget -c -q --tries=20 -P "$RAW" https://ml-site.cdn-apple.com/datasets/cavqa/val/cavqa_val.tar.gz
  tar xzf "$RAW/cavqa_val.tar.gz" -C "$RAW" && rm "$RAW/cavqa_val.tar.gz"
  VAL=$(dirname "$(find "$RAW" -maxdepth 3 -name dataset_dict.json | head -1)")
  python "$REPO/preprocessing/cavqa_convert.py" val "$VAL" "$OUT" --peek
  python "$REPO/preprocessing/cavqa_convert.py" val "$VAL" "$OUT" --depth
  touch "$OUT/.val_done"
fi

for t in ${TASKS:-binary multichoice regression cardinality}; do
  [ -f "$OUT/.${t}_done" ] && continue
  echo "== $t: download"
  grep -v '^$' "$LINKS/train_$t.txt" | xargs -P 16 -I{} wget -c -q --tries=20 -x -nH --cut-dirs=3 -P "$RAW" {}
  python - "$RAW/cavqa_$t/1.0.0" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); info = json.load(open(d / "dataset_info.json"))["splits"][0]
files = list(d.glob("*.tfrecord-*")); size = sum(f.stat().st_size for f in files)
frames = sum(map(int, info["shardLengths"]))
want = int(info["numBytes"]) + 16 * frames  # numBytes leaves out the 16 bytes of framing per record
print(f"{d.parent.name}: {len(files)} / {len(info['shardLengths'])} shards, {size} / {want} bytes")
assert len(files) == len(info["shardLengths"]) and size == want, "incomplete: run the script again (wget -c resumes)"
EOF
  python "$REPO/preprocessing/cavqa_convert.py" train "$RAW" "$OUT" --tasks=$t --peek
  pids=()
  for i in $(seq 0 $((NP-1))); do
    python "$REPO/preprocessing/cavqa_convert.py" train "$RAW" "$OUT" --tasks=$t --part=$i/$NP --depth > "$OUT/logs/$t.$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done  # set -e stops here if a part failed
  grep -h " kept " "$OUT"/logs/$t.*.log | grep skipped | awk -v t=$t '{k+=$3; a+=$5} END {printf "%s kept %d / %d (%.2f%%)\n", t, k, a, 100*k/a; exit (a==0 || k/a<0.95)}'
  [ "${DELETE_RAW:-0}" = 1 ] && rm -rf "$RAW/cavqa_$t"
  touch "$OUT/.${t}_done"
done
