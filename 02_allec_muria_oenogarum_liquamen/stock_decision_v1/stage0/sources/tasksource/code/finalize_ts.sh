#!/bin/sh
# stock_decision_v1 Stage 0 tasksource: inventory, conversion manifest, uploads, read-back md5 check, then _READY.json LAST.
set -e
W=/data/local/sdv1/ts; cd $W
S=s3://YOUR_BUCKET/jev-race/curation/stock_decision_v1/sources/tasksource_ff_v1
nice -n 10 python3 code/inventory_ts.py tables/units_tasksource_ff_v1.jsonl.gz tokout2/units_tokens_tasksource_ff_v1.jsonl.gz meta_hist/sources.yaml tasksource_sources_v1.json > inventory.out
tail -c 3000 inventory.out
python3 - <<'PY'
import json, hashlib, os
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
man=json.load(open('pool/tasksource_ff_v1/manifest.json'))
code={f: {"sha256": sha('code/'+f), "md5": md5('code/'+f)} for f in ("convert_tasksource_ff.py","unittable_ts.py","tokcount.py","tokcount_parity_only.py","validate_ts_pool.py","inventory_ts.py","trainers/claude_lmscore.py","trainers/cache_features.py")}
tokpin=json.load(open('tok/TOKENIZER_PIN.json')) if os.path.exists('tok/TOKENIZER_PIN.json') else None
val=[json.loads(l) for l in open('validate_ts_pool.json') if l.startswith('{')][-1]
tsum=json.load(open('tokout2/summary_tasksource_ff_v1.json'))
par=json.load(open('tokout2/parity_tasksource_ff_v1.json'))
conv={"schema":"stock-decision-tasksource-conversion-manifest-v1","pool":"tasksource_ff_v1","pin":json.load(open('tasksource_pin_v1.json'))["pinned_data_revision"],
 "converter_manifest":man,"code":code,"tokenizer":{"repo":"google/gemma-4-31B-it","revision":"842da3794eaa0b77d5f08bae87a17459d91ff475","pin_file":tokpin},
 "reader_validation":val,"units_table":{"file":"units_tasksource_ff_v1.jsonl.gz","md5":md5('tables/units_tasksource_ff_v1.jsonl.gz'),"summary":json.load(open('tables/units_tasksource_ff_v1.summary.json'))["totals"]},
 "tokens_table":{"file":"units_tokens_tasksource_ff_v1.jsonl.gz","md5":md5('tokout2/units_tokens_tasksource_ff_v1.jsonl.gz'),
   "note":"tok_r1 / tok_r2 = untruncated Gemma prompt tokens (chat template included) under production build_prompt (scorer e0672bd0), stored option order; R2 = noul_defs on (tasksource nouls carry no definitions, so R2 == R1 here); K > 52 units are not rendered (tok null, k_gt_52_unrendered)"},
 "token_parity":par, "tokcount_runs":"main pass: tokcount.py md5 53733f6c (v2 tool + K>52 rows written unrendered); it crashed in the parity SAMPLER on those rows after writing the table/summary/inputs (inputs.tool_sha256 is that version). Parity completed by tokcount_parity_only.py with tokcount.py md5 bf2c9631 (sampler skips K>52 rows; nothing else changed).","tgz":{"file":"tasksource_ff_v1.tgz","md5":md5('tasksource_ff_v1.tgz'),"bytes":os.path.getsize('tasksource_ff_v1.tgz')},
 "inventory":{"file":"tasksource_sources_v1.json","md5":md5('tasksource_sources_v1.json')}}
json.dump(conv, open('conversion_manifest_tasksource_ff_v1.json','w'), indent=1, sort_keys=True, default=str)
print('parity', json.dumps(par)[:600])
PY
for f in tokout2/units_tokens_tasksource_ff_v1.jsonl.gz tokout2/summary_tasksource_ff_v1.json tokout2/parity_tasksource_ff_v1.json tokout2/inputs_tasksource_ff_v1.json \
         tasksource_sources_v1.json conversion_manifest_tasksource_ff_v1.json pool/rejects_tasksource_ff_v1.jsonl pool/tasksource_ff_v1/manifest.json; do
  b=$(basename $f); [ "$b" = manifest.json ] && b=pool_manifest.json
  aws s3 cp --only-show-errors $f $S/$b
done
aws s3 cp --only-show-errors pool/tasksource_ff_v1/counts.json $S/pool_counts.json
for f in code/convert_tasksource_ff.py code/unittable_ts.py code/tokcount.py code/tokcount_parity_only.py code/validate_ts_pool.py code/inventory_ts.py code/finalize_ts.sh; do aws s3 cp --only-show-errors $f $S/code/$(basename $f); done
# read-back md5 of every object under the prefix (except _READY.json)
aws s3 ls --recursive $S/ | awk '{print $4}' | grep -v '_READY.json$' | sort > objects.txt
: > readback.txt
while read k; do m=$(aws s3 cp --only-show-errors s3://YOUR_BUCKET/$k - | md5sum | cut -c1-32); sz=$(aws s3 ls s3://YOUR_BUCKET/$k | awk '{print $3}'); echo "$k $m $sz" >> readback.txt; done < objects.txt
python3 - <<'PY'
import json, hashlib, os
loc={}
pairs={'raw/filtered-full_975339b3/':'v3/','raw/filtered-full_e4d7c06b_report_verified/':'v1/'}
def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
local_map={'tasksource_ff_v1.tgz':'tasksource_ff_v1.tgz','units_tasksource_ff_v1.jsonl.gz':'tables/units_tasksource_ff_v1.jsonl.gz',
 'units_tasksource_ff_v1.summary.json':'tables/units_tasksource_ff_v1.summary.json','units_tokens_tasksource_ff_v1.jsonl.gz':'tokout2/units_tokens_tasksource_ff_v1.jsonl.gz',
 'summary_tasksource_ff_v1.json':'tokout2/summary_tasksource_ff_v1.json','parity_tasksource_ff_v1.json':'tokout2/parity_tasksource_ff_v1.json',
 'inputs_tasksource_ff_v1.json':'tokout2/inputs_tasksource_ff_v1.json','tasksource_sources_v1.json':'tasksource_sources_v1.json',
 'conversion_manifest_tasksource_ff_v1.json':'conversion_manifest_tasksource_ff_v1.json','tasksource_pin_v1.json':'tasksource_pin_v1.json',
 'rejects_tasksource_ff_v1.jsonl':'pool/rejects_tasksource_ff_v1.jsonl','pool_manifest.json':'pool/tasksource_ff_v1/manifest.json',
 'pool_counts.json':'pool/tasksource_ff_v1/counts.json','validate_ts_pool.json':'validate_ts_pool.json'}
pre='jev-race/curation/stock_decision_v1/sources/tasksource_ff_v1/'
objs={}; bad=[]
for l in open('readback.txt'):
    k,m,sz=l.split()
    rel=k[len(pre):]
    lp=None
    for a,b in pairs.items():
        if rel.startswith(a): lp=b+rel[len(a):]
    if rel.startswith('meta/'): lp=rel[len('meta/'):]
    if rel.startswith('code/'): lp='code/'+rel[len('code/'):]
    lp=lp or local_map.get(rel)
    lm=md5(lp) if lp and os.path.exists(lp) else None
    if lm is not None and lm!=m: bad.append(rel)
    objs[rel]={"md5":m,"bytes":int(sz),"local_md5_match":(lm==m) if lm else None}
ready={"schema":"stock-decision-tasksource-ready-v1","pool":"tasksource_ff_v1","objects":objs,"n_objects":len(objs),"readback_mismatches":bad,
 "status":"READY" if not bad else "MISMATCH"}
json.dump(ready, open('_READY.json','w'), indent=1, sort_keys=True)
print('objects', len(objs), 'mismatch', bad, 'unverified_local', [k for k,v in objs.items() if v['local_md5_match'] is None])
PY
grep -q '"status": "READY"' _READY.json && aws s3 cp --only-show-errors _READY.json $S/_READY.json && echo READY_WRITTEN
md5sum _READY.json tasksource_sources_v1.json conversion_manifest_tasksource_ff_v1.json tasksource_pin_v1.json
