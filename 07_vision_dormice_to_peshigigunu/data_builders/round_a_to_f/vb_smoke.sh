#!/bin/bash
# Smoke package: 96 rows stratified across sources from arm_a_nc + only their images -> S3 recipe_a/smoke/ (md5 read-back).
set -u
D=/data/local/visdata; S3=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_a; L=$D/logs/package.log
log(){ echo "[smoke-pkg $(date -u +%H:%M:%SZ)] $*" >> $L; }
mkdir -p $D/pkg/smoke
python3 - <<'PY'
import json, random, collections
rows = [json.loads(l) for l in open('/data/local/visdata/manifests/arm_a_nc.jsonl')]
by = collections.defaultdict(list)
for r in rows:
    by[r['source']].append(r)
rng = random.Random(7)
pick = []
per = max(1, 96 // len(by))
for s in sorted(by):
    pick += rng.sample(by[s], min(per, len(by[s])))
rng.shuffle(pick)
with open('/data/local/visdata/pkg/smoke/smoke.jsonl', 'w') as f:
    for r in pick[:96]:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open('/data/local/visdata/pkg/smoke/images.txt', 'w') as f:
    for p in sorted({p for r in pick[:96] for p in r['images']}):
        f.write(p + '\n')
print(len(pick[:96]), collections.Counter(r['source'] for r in pick[:96]))
PY
tar cf $D/pkg/smoke/smoke_images.tar -C $D/img -T $D/pkg/smoke/images.txt
for f in smoke.jsonl smoke_images.tar; do
  aws s3 cp --quiet $D/pkg/smoke/$f $S3/smoke/$f
  a=$(md5sum < $D/pkg/smoke/$f | cut -d" " -f1); b=$(aws s3 cp --quiet $S3/smoke/$f - | md5sum | cut -d" " -f1)
  [ "$a" = "$b" ] && log "S3_OK smoke/$f md5=$a" || log "S3_MD5_BAD smoke/$f"
done
log SMOKE_PKG_DONE
