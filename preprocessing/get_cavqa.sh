#!/bin/bash
# Download CA-VQA (val + 4 train tasks) one task at a time, convert, delete raw. Peak disk ~2.7 TB.
# Usage: RAW=/scratch/cavqa_raw OUT=/scratch/cavqa_records bash preprocessing/get_cavqa.sh
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
print(f"{d.parent.name}: {len(files)} / {len(info['shardLengths'])} shards, {size/1e12:.3f} / {int(info['numBytes'])/1e12:.3f} TB")
assert len(files) == len(info["shardLengths"]), "missing shards: run the script again (wget -c resumes)"
EOF
  python "$REPO/preprocessing/cavqa_convert.py" train "$RAW" "$OUT" --tasks=$t --peek
  for i in $(seq 0 $((NP-1))); do
    python "$REPO/preprocessing/cavqa_convert.py" train "$RAW" "$OUT" --tasks=$t --part=$i/$NP --depth > "$OUT/logs/$t.$i.log" 2>&1 &
  done
  wait
  grep -h "kept" "$OUT"/logs/$t.*.log
  rm -rf "$RAW/cavqa_$t"
  touch "$OUT/.${t}_done"
done
