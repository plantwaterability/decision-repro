#!/bin/bash
# kl_run.sh ARM (KL agent; prereg "PREREG KL" in AGENT_NOTES). Node-side, idempotent, resumable; S3 markers under s3 .../esbar_kl_<arm>/markers/.
#   k1   = Apache-clean line: init munku (identity 6a0b6e48; NEVER karpo's weights), karpo layout, FS5 gold + karpo OUTPUT probabilities (alpha 0.8)
#   k2   = karpo-LICENSED line (a derivative of karpo-decision-31B, MachinePulse Community License): init karpo's weights, FS5 gold only, lr 1e-6
#   p1e6 / p5e6 = K1's lr probes (200 steps, nothing kept but the dev evals)
# k1: steps 0-200 at 2.5e-6 (= its own probe; saved), then the two probes, then kl_lr_choice.py fixes the lr (LR_CHOICE on S3), then legs.
# Legs: -> MID (export, 4,400 screen in karpo's engine) -> FULL (export, screen, the 31,198-row prefix).
set -u; source /data/local/kl/scripts/kl_env.sh
ARM=${1:?}; STEPS=6574; MID=3287; PROBE=200; SEED=1; DATA=/data/local/kl/data/fs5k      # k1e2 sets its own seed, k7 its own data / steps below
MUNKU_ID=6a0b6e48b55a950ee1f739af7c938cea5b16ec01ef2b233a033640f9a15f8db9; KARPO_ID=70f785365c81a4d8
case $ARM in
  k1) INIT=$MUNKU; ALPHA=0.8; LR=2.5e-6; OA=k1; NOTE="K1 Apache-clean line: init munku; trained on FS5 gold + karpo-decision-31B OUTPUT probabilities only; holds no karpo weights";;
  p1e6) INIT=$MUNKU; ALPHA=0.8; LR=1e-6; OA=k1; NOTE=probe;;
  p5e6) INIT=$MUNKU; ALPHA=0.8; LR=5e-6; OA=k1; NOTE=probe;;
  k1b) INIT=$MUNKU; ALPHA=0.8; LR=2.5e-6; OA=k1b; NOTE="K1 Apache-clean line: init munku; trained on FS5 gold + karpo-decision-31B OUTPUT probabilities only; holds no karpo weights";;      # = K1 exactly at lr 2.5e-6, continued from K1's kept 200 steps (the note string is part of the run identity: it must equal K1's)
  k3) INIT=$K/src/mj50; ALPHA=0.8; LR=1e-6; OA=k3; NOTE="K3 Apache-clean line: init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 (Apache-2.0), identity 97887e20; trained on FS5 gold + karpo-decision-31B OUTPUT probabilities only; holds no karpo weights";;
  k7|k8|k9|k10|k11|k12|k13|k14|k15|k16|k17|k18|k19|k20|k21|k22|k23|k24|k7e2)      # the pure-distillation family (coordinator orders 19:05Z / 22:05Z 11-15): only the 359,747 rows with a karpo target, KL term alone (alpha 1: no gold CE on any row)
      INIT=$K/src/mj50; ALPHA=1.0; LR=1e-6; OA=$ARM; DATA=$K/data/fs5kt; STEPS=5622; MID=2811; VAR="K7's recipe"
      case $ARM in k8) SEED=2; VAR="K7's recipe, seed 2";; k9) LR=2.5e-6; VAR="K7's recipe, lr 2.5e-6";; k10) LR=5e-7; VAR="K7's recipe, lr 5e-7";; k12) SEED=3; VAR="K7's recipe, seed 3";;
        k17) SEED=6; STEPS=2811; MIDS=none; SHORT=1; VAR="K7's recipe, seed 6, a 2,811-step schedule: warm-up 100, cosine over 2,811, the first 2,811 steps of the seed's row order";;
        k18) SEED=7; STEPS=2811; MIDS=none; SHORT=1; VAR="K7's recipe, seed 7, a 2,811-step schedule: warm-up 100, cosine over 2,811, the first 2,811 steps of the seed's row order";;
        k13) SEED=5; EPOCHS=2.0; STEPS=11244; MIDS="2811 5622 8433"; VAR="K7's recipe as ONE two-pass schedule: 11,244 steps, one warm-up + one cosine, rows reshuffled for the second pass, seed 5";; esac
      NOTE="K7 Apache-clean line: init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 (Apache-2.0), identity 97887e20; PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities (359,747 FS5 rows, KL term alone, no gold); holds no karpo weights"
      case $ARM in k7) ;; k8|k9|k10|k12|k13|k17|k18) NOTE="$(echo "$NOTE" | sed "s/^K7 /$(echo $ARM | tr k K) /") ($VAR)";;
        k11) INIT=$MUNKU; NOTE="K11 Apache-clean line: init munku alone (identity 6a0b6e48; all-ours weights); PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities (359,747 FS5 rows, KL term alone, no gold); holds no karpo weights";;
        k14) ENS=1; NOTE="K14 Apache-clean line: init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 (Apache-2.0), identity 97887e20; PURE distillation: trained on karpo-decision-31B and Jeffan-27B OUTPUT probabilities only (the equal-weight average of the two teachers' output probabilities on 359,747 FS5 rows, KL term alone, no gold); holds no karpo and no Jeffan-27B weights";;
        k21) LR=5e-7; HH=fs5hh; MAXROW=16384; NOTE="K21 Apache-clean line: init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 (Apache-2.0), identity 97887e20; PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities (FS5 rows incl. round H0 (chess, bv1_discovery left out) + round-H pools HellaSwag / SATA-MultiRC / GSM8K answer-groups / BANKING77 / SATA-GoEmotions (cfcolor, NC, left out), minus DH's round-H drop list; KL term alone, no gold); holds no karpo weights (K7's recipe at lr 5e-7, one pass)";;
        k23) INIT=$K/src/soup_k7k8k12k3; LR=5e-7; HH=fs5hn; MIDS=none; SAVEEVERY=200; MAXROW=16384; LW=10800; NOTE="K23 Apache-clean line: init soup_k7k8k12k3 (identity 7ecdd654) = quarters of K7 mid, K8 mid, K12 final, K3 final (each from init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 Apache-2.0, trained only on FS5 gold and karpo-decision-31B OUTPUT probabilities); PURE distillation on the round H0 + H rows (cfcolor, NC, left out) + an equal-size fs5kt replay: trained only on karpo-decision-31B OUTPUT probabilities; holds no karpo weights";;
        k22) INIT=$K/src/soup_k7k8k12k3; LR=5e-7; HH=fs5hnc; MIDS=none; SAVEEVERY=200; MAXROW=16384; LW=10800; PREFULL=k23; SELFFULL=73.95; NOTE="K22 candidate line (Apache-2.0 weights; sources listed on the card, incl. the NC-SA one, house precedent confirmed by the user 11-16): K23's data + the 4,841 cfcolor rows (CC BY-NC-SA 2.5 CA: NonCommercial + ShareAlike source); init soup_k7k8k12k3 (identity 7ecdd654) = quarters of K7 mid, K8 mid, K12 final, K3 final (each from init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 Apache-2.0); PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities; holds no karpo weights";;
        k24) INIT=$K/src/soup_k7k8k12k3; LR=5e-7; HC=fs5hc; MIDS=none; SAVEEVERY=200; MAXROW=16384; POSTSTEP=kl_k24post.sh; NOTE="K24 Apache-clean line: init soup_k7k8k12k3 (identity 7ecdd654) = quarters of K7 mid, K8 mid, K12 final, K3 final (each from init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 Apache-2.0, trained only on FS5 gold and karpo-decision-31B OUTPUT probabilities); PURE distillation on the round H0 rows + the round-H pools SATA-MultiRC / SATA-GoEmotions / GSM8K answer-groups / BANKING77 (HellaSwag and cfcolor left out) + an equal-size fs5kt replay: trained only on karpo-decision-31B OUTPUT probabilities; holds no karpo weights";;      # TA agent, coordinator ~22:30Z 11-16: K23's recipe exactly, minus the HellaSwag pool (fs5hc, kl_hbuild24.sh)
        k20) LR=5e-7; HX=1; MAXROW=16384; NOTE="K20 Apache-clean line: init mx_mj50 = 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 (Apache-2.0), identity 97887e20; PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities (FS5 rows with a karpo target incl. round H0's relabelled rows, H0 chess and bv1_discovery rows left out; KL term alone, no gold); holds no karpo weights (K7's recipe at lr 5e-7, one pass)";;
        k19) INIT=$K/src/msj33; NOTE="K19 Apache-clean line: init mx_o_msj33 = thirds of munku / sthanika-ai/Sieve-31B-v3 (Apache-2.0) / tattrongvu/Jeffan-31b-1710 (Apache-2.0), the MX merge, identity 518ed8de; PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities (359,747 FS5 rows, KL term alone, no gold); holds no karpo weights";;
        k15) INIT=$K/src/jeffan; NOTE="K15 Apache-clean line: init tattrongvu/Jeffan-31b-1710 alone (Apache-2.0; the MX source copy, identity 774e5b69); PURE distillation: trained only on karpo-decision-31B OUTPUT probabilities (359,747 FS5 rows, KL term alone, no gold); holds no karpo weights";;
        k16) INIT=$K/src/jeffan; ENS=1; NOTE="K16 Apache-clean line: init tattrongvu/Jeffan-31b-1710 alone (Apache-2.0; the MX source copy, identity 774e5b69); PURE distillation: trained on karpo-decision-31B and Jeffan-27B OUTPUT probabilities only (the equal-weight average of the two teachers' output probabilities on 359,747 FS5 rows, KL term alone, no gold); holds no karpo and no Jeffan-27B weights";;
        k7e2) INIT=$K/src/k7_full; SEED=4; NOTE="K7e2 Apache-clean line: second pure-distillation pass (seed 4) from K7's final weights (K7 = init mx_mj50, 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 Apache-2.0, trained only on karpo-decision-31B OUTPUT probabilities); holds no karpo weights";;
      esac;;
  k1e2) INIT=$K/src/k1_full; ALPHA=0.8; LR=1e-6; OA=k1e2; SEED=2; NOTE="K1e2 Apache-clean line: second pass (new row order, seed 2) from K1's final weights (K1 = init munku + FS5 gold + karpo-decision-31B OUTPUT probabilities); holds no karpo weights";;
  k5) INIT=$K/src/km_t30; ALPHA=0.8; LR=1e-6; OA=k5; SPEC="karpo:1,munku:0.3,stock:-0.3"; WANT=3e3eea8d90d07091e6ea33e77b0242fe7fd573a79074a9ea5aeb205bccf9c84b; EXPECT=75.54
      NOTE="K5 karpo-LICENSED line: a DERIVATIVE of MachinePulseAI/karpo-decision-31B (init = karpo + 0.3 x (munku - stock), the text agent's mx_s_km_t30, identity 3e3eea8d; then FS5 gold + karpo output probabilities): MachinePulse Community License applies; publishing is the user's decision";;
  k6) INIT=$K/src/km50; ALPHA=0.8; LR=1e-6; OA=k6; SPEC="karpo:0.5,munku:0.5"; WANT=964d03733bc4058d505ebe7159a7ab627cda09cde758302784ed31aaa608b3f5; EXPECT=74.66
      NOTE="K6 karpo-LICENSED line: a DERIVATIVE of MachinePulseAI/karpo-decision-31B (init = 0.5 karpo + 0.5 munku, the text agent's mx_s_km50, identity 964d0373; then FS5 gold + karpo output probabilities): MachinePulse Community License applies; publishing is the user's decision";;
  k2) INIT=$KARPO; ALPHA=0; LR=1e-6; OA=k2; NOTE="K2 karpo-LICENSED line: a DERIVATIVE of MachinePulseAI/karpo-decision-31B (initialised from its weights): MachinePulse Community License applies; publishing is the user's decision";;
  *) echo "ARM k1 | k1b | k1e2 | k2 | k3 | k5 | k6 | k7 | k7e2 | k8..k24 | p1e6 | p5e6"; exit 1;;
esac
NODE=$(cat $K/NODE 2>/dev/null || echo unknown); OWN=$(aws s3 cp --only-show-errors $R/esbar_kl_code/guard/owner_$ARM.txt - 2>/dev/null | awk '{print $1}')
[ -n "$OWN" ] && [ "$OWN" != "$NODE" ] && { echo "[kl_run $ARM $(date -u +%FT%TZ)] arm $ARM is owned by node $OWN (this node: $NODE): nothing to do here"; exit 0; }      # one node per arm (coordinator order 14:15Z 11-15)
O=$R/esbar_kl_$OA; F=$K/run/$ARM; mkdir -p $F/logs; L=$F/logs/chain.log
log() { echo "[kl_run $ARM $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/logs/chain_$ARM.log 2>/dev/null; }
die() { log "KL_RUN_FAILED $*"; exit 1; }
mark() { echo "$(date -u +%FT%TZ) $(hostname) ${2:-}" | aws s3 cp --only-show-errors - $O/markers/$1; }
has() { aws s3 ls $O/markers/$1 > /dev/null 2>&1; }
exec 8> $K/node.lock; if [ "${LW:-0}" -gt 0 ]; then flock -w $LW 8; else flock -n 8; fi || { echo "NODE_LOCK held: another kl job runs"; exit 3; }      # LW: an arm queued behind another kl job on its node waits up to LW seconds
[ "$NODE" = p10 ] || echo "whole node: KL agent $ARM (karpo-layout trainer), (re)launched $(date -u +%FT%TZ)" > /tmp/claim.txt      # p10 (the MX agent's node): the claim note is written only when its GPUs are taken
[ -f $K/.staged ] || die NOT_STAGED
[ -s $K/data/fs5k.info.json ] && [ -s $K/data/devk.info.json ] || { for f in fs5k.ids.npy fs5k.meta.npz fs5k.info.json devk.ids.npy devk.meta.npz devk.info.json kl_data.SHA256SUMS; do aws s3 cp --only-show-errors $R/esbar_kl_data/$f $K/data/$f || die "DATA $f"; done; (cd $K/data && sha256sum -c --quiet kl_data.SHA256SUMS 2>/dev/null | grep -v rowids) && die DATA_SHA; }
grep -q 7e6ac6b00e7182337aaceb5786f0e560b5f5410454d8b90b1079fdd228cc578d $K/data/fs5k.info.json || die DATA_IDENTITY
if [ "$DATA" = "$K/data/fs5kt" ]; then DATA=$K/data/fs5kt      # the teacher-row subset of fs5k (kl_subset.py), built once, kept on S3 for restarts
  if [ ! -s $DATA.info.json ]; then
    if aws s3 ls $R/esbar_kl_data/fs5kt.SHA256SUMS > /dev/null 2>&1; then for f in fs5kt.ids.npy fs5kt.meta.npz fs5kt.info.json fs5kt.SHA256SUMS; do aws s3 cp --only-show-errors $R/esbar_kl_data/$f $K/data/$f || die "DATA $f"; done; (cd $K/data && sha256sum -c --quiet fs5kt.SHA256SUMS) || die DATA_SHA_FS5KT
    else $V/bin/python $SC/kl_subset.py $K/data/fs5k $DATA > $F/logs/subset.out 2>&1 || die "SUBSET: $(tail -1 $F/logs/subset.out | cut -c1-200)"
      (cd $K/data && sha256sum fs5kt.ids.npy fs5kt.meta.npz fs5kt.info.json > fs5kt.SHA256SUMS); for f in fs5kt.ids.npy fs5kt.meta.npz fs5kt.info.json fs5kt.SHA256SUMS; do aws s3 cp --only-show-errors $K/data/$f $R/esbar_kl_data/$f || die "DATA_UP $f"; done
    fi
  fi
  grep -q '"rows": 359747' $DATA.info.json && grep -q '"teacher_rows": 359747' $DATA.info.json || die DATA_FS5KT_ROWS
  log "data: $(tr -d '\n' < $DATA.info.json | cut -c1-260)"
fi
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
getinit() {   # $1 s3 export dir, $2 wanted identity: weights only into $INIT, identity-checked (the allow-list for this arm is exactly $2)
  [ ${#2} = 64 ] || die "INIT_WANTED_IDENTITY [$2]"
  if [ ! -f $INIT/.ok ] || [ "$(cat $INIT/.identity 2>/dev/null)" != "$2" ]; then
    mkdir -p $INIT; aws s3 sync --only-show-errors --exclude 'vbll_head*' --exclude '*.pt' --exclude 'READBACK_MD5.txt' --exclude 'decision_config.json' --exclude 'readout.safetensors' $1/ $INIT/ || die "INIT_SYNC $1"
    local id=$(ident $INIT); [ "$id" = "$2" ] || die "INIT_IDENTITY $id != $2"; echo $id > $INIT/.identity; touch $INIT/.ok
  fi
  [ "$(sha256sum < $INIT/tokenizer.json | cut -c1-64)" = "$(sha256sum < $MUNKU/tokenizer.json | cut -c1-64)" ] || die INIT_TOKENIZER_DIFFERS      # the token data was built with munku's tokenizer
}
case $ARM in
  k2) [ -f $KARPO/.ok ] || bash $M/scripts/mx_get.sh karpo > $F/logs/get_karpo.out 2>&1; [ "$(cut -c1-16 $KARPO/.identity)" = $KARPO_ID ] || die INIT_NOT_KARPO;;
  k19) getinit $R/esbar_mx_o_msj33/export "$(aws s3 cp --only-show-errors $R/esbar_mx_o_msj33/export_meta/soup_identity.txt - | grep -oE '[0-9a-f]{64}' | head -1)"      # MX's exported three-way merge (spec munku:0.3333334,sieve:0.3333333,jeffan:0.3333333; export identity = the merge's)
      [ "$(cat $INIT/.identity)" = 518ed8debb39f5d78669c111eb7b9f85881473cee7f6590992157da20b4f7e9b ] || die INIT_NOT_MSJ33;;
  k15|k16) getinit $R/esbar_mx_src/jeffan 774e5b698223278f2e1c402a591946e9b0bfae562e03a71d02c9fcb84b03ce33;;      # jeffan-31b alone: the identity the MX merges record for this member (esbar_mx_mj50/merge/spec.txt)
  k7e2) has EXPORT_export_full || [ -n "$(aws s3 cp --only-show-errors $R/esbar_kl_k7/export_full.identity.txt - 2>/dev/null | cut -c1-64)" ] || die "K7E2 no final export of K7 yet"
      getinit $R/esbar_kl_k7/export_full "$(aws s3 cp --only-show-errors $R/esbar_kl_k7/export_full.identity.txt - | cut -c1-64)";;
  k5|k6) has STOPPED && { log "stopped earlier (start screen off): nothing to do"; exit 0; }      # karpo derivatives: a karpo / munku merge as the init (coordinator order 17:45Z 11-15)
      [ -f $KARPO/.ok ] || bash $M/scripts/mx_get.sh karpo > $F/logs/get_karpo.out 2>&1; [ "$(cut -c1-16 $KARPO/.identity)" = $KARPO_ID ] || die INIT_NOT_KARPO
      [ "$(cat $MUNKU/.identity)" = "$MUNKU_ID" ] || die MEMBER_NOT_MUNKU
      if [ ! -f $INIT/.ok ] || [ "$(cat $INIT/.identity 2>/dev/null)" != $WANT ]; then
        if [ "$(aws s3 cp --only-show-errors $O/init.identity.txt - 2>/dev/null | cut -c1-64)" = $WANT ]; then      # built before (this arm's own S3 copy)
          mkdir -p $INIT; aws s3 sync --only-show-errors $O/init/ $INIT/ || die INIT_SYNC
        else      # build it as the text agent did: mx_merge.py over E1's layout, fp32 math, one cast
          E1D=$D/soupB_in/e1_export
          if [ ! -f $E1D/.ok ]; then mkdir -p $E1D && aws s3 sync --only-show-errors $E/runs/esbar_v5/export/ $E1D/ || die E1_SYNC; [ "$(ident $E1D)" = c92e70ea0dcbfb4d62e04164e7fe48cff4b292b6b47175ae94b4f5987f63eeb3 ] || die E1_IDENTITY; touch $E1D/.ok; fi
          arg="karpo=$KARPO:$(echo $SPEC | tr ',' '\n' | grep '^karpo:' | cut -d: -f2),munku=$MUNKU:$(echo $SPEC | tr ',' '\n' | grep '^munku:' | cut -d: -f2)"
          if echo $SPEC | grep -q stock; then ST=$D/soup75_in/stock
            if [ ! -f $ST/.ok ]; then mkdir -p $ST && aws s3 sync --only-show-errors $S3/models/gemma-4-31B-it_842da379/ $ST/ || die STOCK_SYNC; i=$(ident $ST); case $i in 4d59c3ca50908a67*) echo $i > $ST/.identity; touch $ST/.ok;; *) die "STOCK_IDENTITY $i";; esac; fi
            arg="$arg,stock=$ST:$(echo $SPEC | tr ',' '\n' | grep '^stock:' | cut -d: -f2)"; fi
          [ -e $INIT ] && mv $INIT $INIT.partial_$(date -u +%d%H%M%S)
          log "merging $SPEC ($(md5sum < $M/scripts/mx_merge.py | cut -c1-8))"; $V/bin/python $M/scripts/mx_merge.py $E1D $INIT $F/logs/merge_stats.json "$arg" > $F/logs/merge.log 2>&1 || die "MERGE: $(grep -E 'STRUCT|Error|assert' $F/logs/merge.log | tail -1 | cut -c1-200)"
          [ "$(ls $INIT/model-*.safetensors | wc -l)" = 12 ] && [ "$(grep -c SPOT_OK $F/logs/merge.log)" = 3 ] || die MERGE_INCOMPLETE
          rm -f $INIT/vbll_head.json $INIT/vbll_head.safetensors      # E1's head files ride along in the layout; not part of this model
        fi
        id=$(ident $INIT); [ "$id" = "$WANT" ] || die "INIT_IDENTITY $id != $WANT (the text agent's merge)"; echo $id > $INIT/.identity; touch $INIT/.ok
        if [ "$(aws s3 cp --only-show-errors $O/init.identity.txt - 2>/dev/null | cut -c1-64)" != $WANT ]; then aws s3 sync --only-show-errors --exclude '.*' $INIT/ $O/init/ && echo $id | aws s3 cp --only-show-errors - $O/init.identity.txt; aws s3 cp --only-show-errors $F/logs/merge.log $O/init_merge.log 2>/dev/null; fi
        log "init $SPEC identity $(cut -c1-16 $INIT/.identity) = the text agent's"
      fi
      [ "$(sha256sum < $INIT/tokenizer.json | cut -c1-64)" = "$(sha256sum < $MUNKU/tokenizer.json | cut -c1-64)" ] || die INIT_TOKENIZER_DIFFERS;;
  k22|k23|k24) getinit $R/esbar_kl_soup_k7k8k12k3/export 7ecdd654795a46805c58f1a2eed4539a937ae56f88700ae9c0b9733142d22ed8;;      # the soup's export (post2), identity from its REQUEST / export.identity.txt
  k3|k7|k8|k9|k10|k12|k13|k14|k17|k18|k20|k21) getinit $R/esbar_mx_mj50/export "$(aws s3 cp --only-show-errors $R/esbar_mx_mj50/export_meta/soup_identity.txt - | grep -oE '[0-9a-f]{64}' | head -1)"
      [ "$(cat $INIT/.identity)" = 97887e207344be14208369dc60c3a7224e2093d63e820c05110bbe999343c93b ] || die INIT_NOT_MJ50;;
  k1e2) has DONE && { log "done already"; exit 0; }      # the coordinator's rule (17:10Z 11-15): only if K1's FINAL screen is >= 73.10; else the node is parked
      X=$(grep -E '^INDEX KOWN' $M/work/kl_k1_full/kown_s4400/score.txt 2>/dev/null | tail -1 | awk '{print $3}')
      [ -n "$X" ] || X=$(aws s3 cp --only-show-errors $R/esbar_kl_k1_full/kown_s4400/logs/driver.log - 2>/dev/null | grep -E '^INDEX KOWN' | tail -1 | awk '{print $3}')
      [ -n "$X" ] || die "K1E2_GATE no final screen index of K1 found"
      if ! has GATE; then
        if python3 -c "import sys; sys.exit(0 if float('$X') >= 73.10 else 1)"; then mark GATE "PASS: K1 final screen $X >= 73.10 -> K1e2 runs"; log "GATE PASS: K1 final screen $X >= 73.10"
        else mark SKIPPED "K1 final screen $X < 73.10: K1e2 not run; node parked"; mark DONE "SKIPPED: K1 final screen $X < 73.10 (no second pass); node parked"; log "GATE FAIL: K1 final screen $X < 73.10: K1e2 skipped, node parked"; exit 0; fi
      fi
      has SKIPPED && exit 0
      getinit $R/esbar_kl_k1/export_full "$(aws s3 cp --only-show-errors $R/esbar_kl_k1/export_full.identity.txt - | cut -c1-64)";;
  *)  [ "$(cat $INIT/.identity)" = "$MUNKU_ID" ] && [ "$INIT" = "$MUNKU" ] || die INIT_NOT_MUNKU;;
esac
case $ARM in k2) ;; *) ls $INIT | grep -q "vbll_head\|adapter_\|decision_config\|readout" && die INIT_HAS_FOREIGN_FILES;; esac
ensdata() {   # fs5ke = fs5kt's rows with the ENSEMBLE teacher's targets (s3 .../esbar_mx/tl/teacher_ens_fs5.jsonl.gz: average of karpo's and Jeffan-27B's OUTPUT probabilities); 1 = not there yet
  local W=$K/data f; [ -s $W/fs5ke.info.json ] && return 0
  if aws s3 ls $R/esbar_kl_data/fs5ke.SHA256SUMS > /dev/null 2>&1; then for f in fs5ke.ids.npy fs5ke.meta.npz fs5ke.info.json fs5ke.SHA256SUMS; do aws s3 cp --only-show-errors $R/esbar_kl_data/$f $W/$f || die "DATA $f"; done; (cd $W && sha256sum -c --quiet fs5ke.SHA256SUMS) || die DATA_SHA_FS5KE; return 0; fi
  aws s3 ls $R/esbar_mx/tl/teacher_ens_fs5.jsonl.gz > /dev/null 2>&1 || return 1
  aws s3 cp --only-show-errors $R/esbar_mx/tl/teacher_ens_fs5.jsonl.gz $W/teacher_ens_fs5.jsonl.gz || die TEACHER_ENS_FETCH
  [ -s $W/fs5k.rowids.txt ] || aws s3 cp --only-show-errors $R/esbar_kl_data/fs5k.rowids.txt $W/fs5k.rowids.txt || die ROWIDS; [ -s $W/fs5.jsonl.gz ] || aws s3 cp --only-show-errors $R/esbar_mx/tl/fs5.jsonl.gz $W/fs5.jsonl.gz || die REQUESTS
  $V/bin/python $SC/kl_swapq.py $W/fs5kt $W/fs5k.rowids.txt $W/fs5k.meta.npz $W/fs5.jsonl.gz $W/teacher_ens_fs5.jsonl.gz $W/fs5ke.tmp > $F/logs/swapq.out 2>&1 || die "SWAPQ: $(tail -1 $F/logs/swapq.out | cut -c1-240)"
  for f in ids.npy meta.npz info.json; do mv $W/fs5ke.tmp.$f $W/fs5ke.$f; done; (cd $W && sha256sum fs5ke.ids.npy fs5ke.meta.npz fs5ke.info.json > fs5ke.SHA256SUMS)
  for f in fs5ke.ids.npy fs5ke.meta.npz fs5ke.info.json fs5ke.SHA256SUMS; do aws s3 cp --only-show-errors $W/$f $R/esbar_kl_data/$f || die "DATA_UP $f"; done; aws s3 cp --only-show-errors $F/logs/swapq.out $R/esbar_kl_data/fs5ke.swapq.out 2>/dev/null; return 0; }
hxdata() {   # fs5kh = fs5kt's rows + the H0 rows (karpo's labels for the FS5 rows MX's converter could not parse) minus the sources the pointer excludes (coordinator 17:25Z 11-16)
  # pointer s3 .../esbar_kl_code/guard/k20_inputs.txt: "requests <s3> <sha256>", "teacher <s3> <sha256>", any number of "exclude <source> <exact count>" (H0 rows only). 1 = not there yet.
  local W=$K/data f rq rs te ts ex n0; [ -s $W/fs5kh.info.json ] && return 0
  if aws s3 ls $R/esbar_kl_data/fs5kh.SHA256SUMS > /dev/null 2>&1; then for f in fs5kh.ids.npy fs5kh.meta.npz fs5kh.info.json fs5kh.SHA256SUMS fs5kh.h0sel.json fs5kh.check.json; do aws s3 cp --only-show-errors $R/esbar_kl_data/$f $W/$f || die "DATA $f"; done; (cd $W && sha256sum -c --quiet fs5kh.SHA256SUMS) || die DATA_SHA_FS5KH; return 0; fi
  aws s3 cp --only-show-errors $R/esbar_kl_code/guard/k20_inputs.txt $W/k20_inputs.txt 2>/dev/null || return 1
  read rq rs < <(awk '$1 == "requests" {print $2, $3}' $W/k20_inputs.txt); read te ts < <(awk '$1 == "teacher" {print $2, $3}' $W/k20_inputs.txt)
  ex=$(awk '$1 == "exclude" {printf "%s%s:%s", (n++ ? "," : ""), $2, $3}' $W/k20_inputs.txt)
  [ ${#rs} = 64 ] && [ ${#ts} = 64 ] || die "k20_inputs.txt needs 'requests <s3> <sha256>' and 'teacher <s3> <sha256>' lines"
  aws s3 ls $rq > /dev/null 2>&1 && aws s3 ls $te > /dev/null 2>&1 || return 1
  aws s3 cp --only-show-errors $rq $W/h0_requests.jsonl.gz && aws s3 cp --only-show-errors $te $W/h0_teacher.jsonl.gz || die H0_FETCH
  [ "$(sha256sum < $W/h0_requests.jsonl.gz | cut -c1-64)" = "$rs" ] || die "H0_REQUESTS_SHA256 $(sha256sum < $W/h0_requests.jsonl.gz | cut -c1-16) != ${rs:0:16}"
  [ "$(sha256sum < $W/h0_teacher.jsonl.gz | cut -c1-64)" = "$ts" ] || die "H0_TEACHER_SHA256 $(sha256sum < $W/h0_teacher.jsonl.gz | cut -c1-16) != ${ts:0:16}"
  for f in mxa.jsonl fs5.jsonl.gz teacher_karpo_fs5.jsonl.gz; do [ -s $W/$f ] || aws s3 cp --only-show-errors $R/esbar_mx/tl/$f $W/$f || die "FETCH $f"; done
  [ -s $W/fs5k.rowids.txt ] || aws s3 cp --only-show-errors $R/esbar_kl_data/fs5k.rowids.txt $W/fs5k.rowids.txt || die ROWIDS
  : > $W/fs5kh.exclude_ids.txt; echo '{"excluded": 0, "kept_h0_rows": null}' > $W/fs5kh.h0sel.json
  [ -z "$ex" ] || $V/bin/python $SC/kl_h0sel.py $W/mxa.jsonl $W/h0_requests.jsonl.gz $W/fs5kh.exclude_ids.txt $W/fs5kh.h0sel.json "$ex" > $F/logs/h0sel.out 2>&1 || die "H0SEL: $(tail -1 $F/logs/h0sel.out | cut -c1-400)"
  log "H0 selection ($ex): $(tr -d '\n ' < $W/fs5kh.h0sel.json | cut -c1-700)"
  $V/bin/python $SC/kl_data.py $W/mxa.jsonl $W/fs5khfull $MUNKU $C/trainers --requests $W/fs5.jsonl.gz,$W/h0_requests.jsonl.gz --teacher $W/teacher_karpo_fs5.jsonl.gz,$W/h0_teacher.jsonl.gz \
    --exclude-ids $W/fs5kh.exclude_ids.txt --workers 96 > $F/logs/data_fs5kh.out 2>&1 || die "KLDATA: $(tail -1 $F/logs/data_fs5kh.out | cut -c1-240)"
  $V/bin/python $SC/kl_subset.py $W/fs5khfull $W/fs5kh.tmp > $F/logs/subset_fs5kh.out 2>&1 || die "SUBSET: $(tail -1 $F/logs/subset_fs5kh.out | cut -c1-240)"
  for f in ids.npy meta.npz info.json; do mv $W/fs5kh.tmp.$f $W/fs5kh.$f; done
  $V/bin/python $SC/kl_h0check.py $W/fs5kt $W/fs5k.rowids.txt $W/fs5k.meta.npz $W/fs5khfull $W/fs5kh $W/fs5kh.check.json > $F/logs/h0check.out 2>&1 || log "H0CHECK script failed: $(tail -1 $F/logs/h0check.out | cut -c1-200)"
  log "fs5kt inside fs5kh: $(tr -d '\n ' < $W/fs5kh.check.json 2>/dev/null | cut -c1-400)"; grep -q '"ok": true' $W/fs5kh.check.json 2>/dev/null && mark DATA_CHECK "fs5kt rows unchanged inside fs5kh: $(tr -d '\n ' < $W/fs5kh.check.json | cut -c1-300)" || mark DATA_CHECK "WARNING fs5kt rows NOT all unchanged inside fs5kh (training goes on): $(tr -d '\n ' < $W/fs5kh.check.json 2>/dev/null | cut -c1-300)"
  (cd $W && sha256sum fs5kh.ids.npy fs5kh.meta.npz fs5kh.info.json fs5kh.h0sel.json fs5kh.check.json > fs5kh.SHA256SUMS)
  for f in fs5kh.ids.npy fs5kh.meta.npz fs5kh.info.json fs5kh.h0sel.json fs5kh.check.json fs5kh.exclude_ids.txt fs5khfull.rowids.txt fs5khfull.info.json fs5kh.SHA256SUMS; do aws s3 cp --only-show-errors $W/$f $R/esbar_kl_data/$f || die "DATA_UP $f"; done
  aws s3 cp --only-show-errors $F/logs/data_fs5kh.out $R/esbar_kl_data/fs5kh.kldata.out 2>/dev/null; aws s3 cp --only-show-errors $W/k20_inputs.txt $R/esbar_kl_data/fs5kh.inputs.txt 2>/dev/null; return 0; }
if [ -n "${HX:-}" ]; then w=0; until hxdata; do [ $w = 0 ] && log "waiting for s3 .../esbar_kl_code/guard/k20_inputs.txt (H0 requests + teacher; init and FS5 files are staged)"; w=1; sleep 30; done; DATA=$K/data/fs5kh
  n=$(python3 -c "import json; print(json.load(open('$DATA.info.json'))['rows'])"); kh=$(python3 -c "import json; print(json.load(open('$K/data/fs5kh.h0sel.json')).get('kept_h0_rows') or 0)")
  [ "$n" -ge $(( 359747 + kh * 95 / 100 )) ] || die "DATA_FS5KH rows $n < 359,747 + 95% of the $kh kept H0 rows: see $F/logs/data_fs5kh.out"; STEPS=$(( (n + 63) / 64 )); MID=$(( STEPS / 2 ))
  log "data: fs5kh $n rows -> $STEPS steps, mid $MID | $(grep KLDATA_DONE $F/logs/data_fs5kh.out 2>/dev/null | cut -c1-700) | $(tr -d '\n' < $DATA.info.json | cut -c1-300)"; fi
ENSNOTE="PURE distillation: trained on karpo-decision-31B and Jeffan-27B OUTPUT probabilities only (the equal-weight average of the two teachers' output probabilities on 359,747 FS5 rows, KL term alone, no gold); holds no karpo and no Jeffan-27B weights"
if [ $ARM = k14 ] || [ $ARM = k16 ]; then w=0; until ensdata; do [ $w = 0 ] && log "waiting for s3 .../esbar_mx/tl/teacher_ens_fs5.jsonl.gz (init and fs5kt are staged)"; w=1; sleep 60; done; DATA=$K/data/fs5ke; fi
if [ $ARM = k7e2 ]; then      # coordinator order 23:55Z 11-15: the second pass uses the ensemble teacher; wait up to 45 min for its file, then fall back to karpo-only. The choice is fixed once (marker).
  if ! has DATA_CHOICE; then t0=$(date +%s); log "K7e2: ensemble teacher data wanted (waits up to 45 min from now)"
    until ensdata; do [ $(( $(date +%s) - t0 )) -gt 2700 ] && break; sleep 60; done
    if [ -s $K/data/fs5ke.info.json ]; then mark DATA_CHOICE "fs5ke ensemble teacher (karpo + Jeffan-27B output probabilities)"; else mark DATA_CHOICE "fs5kt karpo-only FALLBACK: no teacher_ens_fs5.jsonl.gz on S3 within 45 min"; fi
  fi
  case "$(aws s3 cp --only-show-errors $O/markers/DATA_CHOICE - | awk '{print $3}')" in
    fs5ke) ensdata || die "K7E2 fs5ke chosen but not there"; DATA=$K/data/fs5ke; NOTE="K7e2 Apache-clean line: second pure-distillation pass (seed 4) from K7's final weights (K7 = init mx_mj50, 0.5 munku + 0.5 tattrongvu/Jeffan-31b-1710 Apache-2.0, trained only on karpo-decision-31B OUTPUT probabilities); this pass: $ENSNOTE";;
    fs5kt) DATA=$K/data/fs5kt;; *) die "K7E2 DATA_CHOICE unreadable";; esac
  log "K7e2 data: ${DATA##*/}"
fi
if [ -n "${HH:-}" ]; then      # round H (coordinator 17:45Z 11-16): fs5hh (K21) / fs5hn (K23) / fs5hnc (K22), built once by kl_hbuild.sh (any node) and fetched elsewhere
  bash $SC/kl_hbuild.sh $HH >> $F/logs/hbuild.out 2>&1 || die "HBUILD $HH: $(tail -1 $K/logs/hbuild.log 2>/dev/null | cut -c1-300)"; DATA=$K/data/$HH
  n=$(python3 -c "import json; print(json.load(open('$DATA.info.json'))['rows'])"); STEPS=$(( (n + 63) / 64 )); MID=$(( STEPS / 2 ))
  log "data: $HH $n rows -> $STEPS steps$([ "${MIDS:-}" = none ] || echo ", mid $MID") | $(tr -d '\n ' < $DATA.info.json | cut -c1-500)"; fi
if [ -n "${HC:-}" ]; then      # K24 (TA agent, coordinator ~22:30Z 11-16): fs5hc = fs5hn's rule without the hellaswag pool, picked from fs5hh by kl_hbuild24.sh / kl_pick24.py (0 hellaswag / cfcolor rows or it stops)
  bash $SC/kl_hbuild24.sh $HC >> $F/logs/hbuild.out 2>&1 || die "HBUILD24 $HC: $(tail -1 $K/logs/hbuild24.log 2>/dev/null | cut -c1-300)"; DATA=$K/data/$HC
  [ "$(python3 -c "import json; p = json.load(open('$DATA.info.json'))['pools']; print(p['hellaswag'] + p['cfcolor'])")" = 0 ] || die "DATA_$HC holds hellaswag / cfcolor rows"
  n=$(python3 -c "import json; print(json.load(open('$DATA.info.json'))['rows'])"); STEPS=$(( (n + 63) / 64 )); MID=$(( STEPS / 2 ))
  log "data: $HC $n rows -> $STEPS steps | $(python3 -c "import json; d = json.load(open('$DATA.info.json')); print({k: d[k] for k in ('rows', 'tokens', 'len_max', 'K_max', 'pools', 'checks', 'data_sha256')})" | cut -c1-900)"; fi
case $DATA in */fs5ke) grep -q '"rows": 359747' $DATA.info.json && grep -q '"ids_equal_to_source": true' $DATA.info.json || die DATA_FS5KE_ROWS; log "data: $(tr -d '\n' < $DATA.info.json | cut -c1-420)";; esac
if [ "$NODE" = p10 ]; then w=0      # p10 is the MX agent's node: train only when its queue is parked (s3 .../esbar_mx/queue/p10.END, or our own .../esbar_kl_code/guard/p10.GO) and the GPUs are idle
  until { aws s3 ls $R/esbar_mx/queue/p10.END > /dev/null 2>&1 || aws s3 ls $R/esbar_kl_code/guard/p10.GO > /dev/null 2>&1; } && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" = 0 ] && sleep 30 && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" = 0 ]; do
    [ $w = 0 ] && log "waiting for p10: MX queue parked (p10.END or p10.GO) and idle GPUs"; w=1; has EXPORT_export_full && break; sleep 60; done
  echo "whole node: KL agent $ARM (p10 lent by the MX agent after its labelling), $(date -u +%FT%TZ)" > /tmp/claim.txt
fi
if [ $ARM = k1 ] && has LR_CHOICE; then LR=$(aws s3 cp --only-show-errors $O/markers/LR_CHOICE - | awk '{print $1}'); fi
RS=$O/resume_lr$LR; RUN=$F/run_lr$LR; mkdir -p $RUN      # resume mirror and run dir per lr (another lr is another run identity)
if [ $ARM = k1b ] && ! has RESUME_COPIED; then      # K1's first 200 steps at 2.5e-6 (kept on S3) become K1b's start; server-side copy, K1's prefix is not touched
  log "copying s3 .../esbar_kl_k1/resume_lr2.5e-6/resume_step200/ -> $RS/resume_step200/"
  aws s3 sync --only-show-errors $R/esbar_kl_k1/resume_lr2.5e-6/resume_step200/ $RS/resume_step200/ || die RESUME_COPY
  a=$(aws s3 ls --recursive --summarize $R/esbar_kl_k1/resume_lr2.5e-6/resume_step200/ | tail -2 | tr '\n' ' '); b=$(aws s3 ls --recursive --summarize $RS/resume_step200/ | tail -2 | tr '\n' ' ')
  [ -n "$a" ] && [ "$a" = "$b" ] || die "RESUME_COPY_SIZE src [$a] dst [$b]"
  mark RESUME_COPIED "$b"; log "resume copied: $b"
fi
train() {   # $1 max steps, $2.. extra flags
  local n=$1; shift
  $TR --rdzv-endpoint 127.0.0.1:29803 $SC/kl_train.py --init-from $INIT --manifest $DATA --val $K/data/devk --val-max 2000 \
    --out $RUN --pinned-code $C/trainers --alpha $ALPHA --lr $LR --warmup 100 --lr-floor 0.1 --weight-decay 0.0 --clip 1.0 \
    --decisions-per-step 64 --sched-steps $STEPS ${EPOCHS:+--epochs $EPOCHS} --max-steps $n --seed $SEED --pack-tokens ${KL_PACK:-8192} --max-row-tokens ${MAXROW:-8192} \
    --eval-every ${KL_EVAL_EVERY:-250} --eval-at-start --licence-note "$NOTE" "$@" >> $F/logs/train.log 2>&1
}
push_logs() { aws s3 cp --only-show-errors $F/logs/train.log $O/logs/train_$ARM.log 2>/dev/null; [ -s $RUN/evals.jsonl ] && aws s3 cp --only-show-errors $RUN/evals.jsonl $O/evals/evals_$ARM.jsonl 2>/dev/null; true; }
( exec 8>&- 7>&- 6>&-; while sleep 600; do push_logs; done ) & PUSHER=$!; trap 'pkill -P $PUSHER 2>/dev/null; kill $PUSHER 2>/dev/null' EXIT      # the pusher (and its sleep) must not hold the node / chain / keeper locks after this script exits (11:13Z: an orphan sleep kept node.lock for 10 min)
last() { grep -E '^STEP' $F/logs/train.log | tail -1 | cut -c1-120; }
# ---- probes: 200 steps, no saves, no export
if [ $ARM = p1e6 ] || [ $ARM = p5e6 ]; then
  has PROBE_$ARM && { log "probe done already"; exit 0; }
  log "PROBE start lr $LR"; train $PROBE --eval-every 100 --save-every 0 --no-export --spot-watch off; rc=$?; push_logs
  [ $rc = 0 ] && [ "$(grep -c . $RUN/evals.jsonl)" -ge 3 ] || die "probe rc=$rc: $(grep -E 'Error|error' $F/logs/train.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
  mark PROBE_$ARM "$(tail -1 $RUN/evals.jsonl)"; log "PROBE_DONE $(tr '\n' ' ' < $RUN/evals.jsonl | cut -c1-400)"; exit 0
fi
SE=100; [ -n "${SHORT:-}" ] && SE=400; [ -n "${SAVEEVERY:-}" ] && SE=$SAVEEVERY      # the short soup seeds save every 400 steps (they must finish before 08:05Z)
SAVE="--save-every $SE --keep-resume 1 --resume-s3 $RS --keep-resume-s3 2 --spot-watch on --force-save-utc 07:35"
# ---- K1 only: its first 200 steps at 2.5e-6 are the 2.5e-6 probe (kept: the run continues from them if the lr stands)
if [ $ARM = k1 ] && ! has LR_CHOICE; then
  if ! has PROBE_k1; then
    log "K1 steps 0-$PROBE at lr $LR (its own lr probe)"; train $PROBE --eval-every 100 --no-export $SAVE; rc=$?; push_logs
    [ $rc = 0 ] || die "k1 probe leg rc=$rc: $(grep -E 'Error|error' $F/logs/train.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
    mark PROBE_k1 "$(tail -1 $RUN/evals.jsonl)"; log "PROBE_k1 done: $(tr '\n' ' ' < $RUN/evals.jsonl | cut -c1-400)"
  fi
  log "K1_WAITS_FOR_LR_CHOICE (probes p1e6 / p5e6, then kl_lr_choice)"; exit 10
fi
upload_export() {   # $1 local dir, $2 s3 name
  local src=$1 dst=$O/$2
  aws s3 sync --only-show-errors $src/ $dst/ || return 1
  (cd $src && ls) | xargs -P 6 -I{} bash -c 'a=$(md5sum < "$0/{}" | cut -c1-32); b=$(aws s3 cp --only-show-errors "$1/{}" - | md5sum | cut -c1-32); [ "$a" = "$b" ] && echo "OK {} $a" || echo "BAD {} $a $b"' $src $dst > $F/logs/readback_$2.txt
  [ "$(grep -c '^BAD ' $F/logs/readback_$2.txt)" = 0 ] && [ "$(grep -c '^OK ' $F/logs/readback_$2.txt)" = "$(ls $src | wc -l)" ] || return 1
  (cd $src && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64 > $F/logs/identity_$2.txt
  aws s3 cp --only-show-errors $F/logs/identity_$2.txt $dst.identity.txt; aws s3 cp --only-show-errors $F/logs/readback_$2.txt $dst/READBACK_MD5.txt
  log "EXPORT_UPLOADED $2 identity $(cut -c1-16 $F/logs/identity_$2.txt) (md5 read-back ok)"
}
screen() {   # $1 export dir, $2 step label, $3 KSET
  local tag=kl_${ARM}_$2 sub=kown_$3
  has READ_${3}_$2 && return 0
  KSET=$3 bash $M/scripts/mx_kown.sh $tag $1 > $F/logs/kown_${3}_$2.out 2>&1; local rc=$?
  local w=$M/work/$tag/kown; [ $3 = prefix ] || w=$M/work/$tag/$sub
  log "READ $3 $2 rc=$rc: INDEX $(grep -E '^INDEX KOWN' $w/score.txt 2>/dev/null | awk '{print $3}') | $(grep -E 'KOWN_DONE|DIE' $w/driver.log | tail -1 | cut -c1-500)"
  [ $rc = 0 ] && mark READ_${3}_$2 "THIS=$(grep -E '^INDEX KOWN' $w/score.txt | awk '{print $3}') | $(grep -E '^INDEX' $w/score.txt | cut -c1-36 | tr '\n' ';' | cut -c1-300)"
}
leg() {   # $1 steps, $2 export name
  local n=$1 name=$2
  if ! has EXPORT_$name; then
    if [ ! -s $RUN/$name/model.safetensors.index.json ]; then
      if [ ! -s $RUN/export/model.safetensors.index.json ]; then
        log "LEG -> $n start (lr $LR, resume $RS)"; t0=$(date +%s); train $n $SAVE; rc=$?; push_logs
        log "LEG -> $n rc=$rc elapsed=$(( ($(date +%s) - t0) / 60 ))min: $(last)"
        [ $rc = 0 ] && [ -s $RUN/export/model.safetensors.index.json ] || die "LEG $n rc=$rc: $(grep -E 'Error|error|STOP_' $F/logs/train.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
      fi
      echo "{\"arm\": \"$ARM\", \"lr\": \"$LR\", \"alpha\": \"$ALPHA\", \"steps\": $n, \"node\": \"$NODE\", \"host\": \"$(hostname)\"}" > $RUN/export/kl_arm.json
      case $ARM in k2|k5|k6) cp $KARPO/LICENSE $KARPO/NOTICE $RUN/export/ || die LICENCE_FILES;; esac      # K2 is a derivative: karpo's licence files travel with it
      mv $RUN/export $RUN/$name || die MV
    fi
    upload_export $RUN/$name $name || die "UPLOAD $name"; mark EXPORT_$name "$(cut -c1-16 $F/logs/identity_$name.txt)"
  fi
  if [ ! -s $RUN/$name/model.safetensors.index.json ]; then mkdir -p $RUN/$name; aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $O/$name/ $RUN/$name/ || die "FETCH $name"; fi
}
case $ARM in k3|k5|k6|k19) screen $INIT s0 s4400;; esac      # the start point: the untrained init in karpo's engine
case $ARM in k5|k6)      # the coordinator's rule: a start screen more than 0.4 off the text agent's number stops the arm
  if ! has START_OK; then
    X=$(grep -E '^INDEX KOWN' $M/work/kl_${ARM}_s0/kown_s4400/score.txt 2>/dev/null | tail -1 | awk '{print $3}')
    [ -n "$X" ] || X=$(aws s3 cp --only-show-errors $R/esbar_kl_${ARM}_s0/kown_s4400/logs/driver.log - 2>/dev/null | grep -E '^INDEX KOWN' | tail -1 | awk '{print $3}')
    [ -n "$X" ] || die "START_SCREEN no index"
    if python3 -c "import sys; sys.exit(0 if abs(float('$X') - $EXPECT) <= 0.4 else 1)"; then mark START_OK "start screen $X (text agent $EXPECT)"; log "START_OK start screen $X (text agent $EXPECT)"
    else mark STOPPED "start screen $X is more than 0.4 off the text agent's $EXPECT: arm stopped before training"; mark DONE "STOPPED: start screen $X vs $EXPECT"; log "ARM_STOPPED start screen $X vs $EXPECT"; exit 0; fi
  fi;; esac
if [ -n "${PREFULL:-}" ] && ! has PREFULL_DONE; then      # coordinator 18:20Z 11-16: on THIS node, before this arm trains: K23's prefix vs the soup per benchmark, and its full public run if its prefix >= 73.95
  PA=$PREFULL; PT=kl_${PA}_full; PO=$R/esbar_kl_$PA; PW=$M/work/$PT/kown; mkdir -p $F/cmp
  until aws s3 ls $PO/markers/DONE > /dev/null 2>&1; do log "PREFULL: waiting for $PA's DONE marker"; sleep 120; done
  for x in "$PW/prefix_compare.json $R/esbar_$PT/kown_prefix/prefix_compare.json $F/cmp/a.json" "$M/work/kl_soup_k7k8k12k3_j2/kown/prefix_compare.json $R/esbar_kl_soup_k7k8k12k3_j2/kown_prefix/prefix_compare.json $F/cmp/b.json" "/nonexistent $R/esbar_kl_soup_k7k8k12k3/kown_prefix/prefix_compare.json $F/cmp/c.json"; do read -r xl xs xo <<< "$x"
    if [ -s $xl ]; then cp $xl $xo; else aws s3 cp --only-show-errors $xs $xo || log "PREFULL: no $xs"; fi; done
  $V/bin/python $SC/kl_cmp.py ${PA^^} $F/cmp/a.json soup_kls $F/cmp/b.json soup_kl3 $F/cmp/c.json $KR/src/prefix_rows.jsonl.gz > $F/cmp/${PA}_vs_soup.txt 2>&1
  aws s3 cp --only-show-errors $F/cmp/${PA}_vs_soup.txt $PO/${PA}_vs_soup.txt; log "PREFULL: $(head -1 $F/cmp/${PA}_vs_soup.txt | cut -c1-260)"
  PP=$(python3 -c "import json; print(json.load(open('$F/cmp/a.json'))['arms']['KOWN']['index'])" 2>/dev/null)
  if [ -n "$PP" ] && python3 -c "import sys; sys.exit(0 if float('$PP') >= 73.95 else 1)"; then
    if aws s3 ls $R/esbar_$PT/kown_full/KFULL.txt > /dev/null 2>&1; then log "PREFULL: $PA's full public run exists already"
    else PD=$K/run/$PA/run_lr$LR/export_full; WANT=$(aws s3 cp --only-show-errors $PO/export_full.identity.txt - | cut -c1-64)
      if [ ! -s $PD/model.safetensors.index.json ]; then PD=$K/src/${PA}_full; mkdir -p $PD; aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $PO/export_full/ $PD/ || die "PREFULL fetch"; fi
      [ "$(ident $PD)" = "$WANT" ] || die "PREFULL identity of $PD"; echo $WANT > $PD/.identity
      log "PREFULL: $PA final prefix $PP >= 73.95: its FULL PUBLIC run on this node now (mx_kfull.sh $PT; nothing public)"
      bash $M/scripts/mx_kfull.sh $PT $PD > $F/logs/prefull_$PA.out 2>&1; rc=$?
      log "PREFULL: mx_kfull.sh rc=$rc: $(grep -E 'KFULL_DONE|DIE' $M/work/$PT/kfull/driver.log 2>/dev/null | tail -1 | cut -c1-300)"
      [ $rc = 0 ] && echo "$(date -u +%FT%TZ) $(hostname) $(grep -E '^HEADLINE' $M/work/$PT/kfull/merge/report.txt 2>/dev/null | cut -c1-300)" | aws s3 cp --only-show-errors - $PO/markers/FULL_DONE
    fi
  else log "PREFULL: $PA final prefix ${PP:-?} < 73.95: no full public run"; fi
  mark PREFULL_DONE "$PA prefix ${PP:-?}; full run $(aws s3 ls $PO/markers/FULL_DONE > /dev/null 2>&1 && echo done || echo none)"
fi
for m in ${MIDS:-$MID}; do [ $m = none ] && continue; leg $m export_s$m; screen $RUN/export_s$m s$m s4400; done      # mid exports + screens (k13: three)
leg $STEPS export_full; screen $RUN/export_full full s4400
if [ $ARM = k8 ] || [ -n "${SHORT:-}" ] || [ -n "${KL_NOPREFIX:-}" ]; then has READ_s4400_full && { mark DONE "final screen read; prefix read left to kl_post.sh (best of K7 / K8 / soup)"; log "KL_RUN_DONE $ARM (no own prefix read)"; }; push_logs; exit 0; fi
screen $RUN/export_full full prefix
if has READ_prefix_full; then PX=$(grep -E '^INDEX KOWN' $M/work/kl_${ARM}_full/kown/score.txt 2>/dev/null | tail -1 | awk '{print $3}'); fi
# the full public run rule lives in kl_guard (s3 .../esbar_kl_code/full/RULE.txt, coordinator 17:45Z 11-16: K20 / K21 / K23 finals with prefix >= 73.95, in order of prefix).
# ARMDONE_<arm> (a terminate_on.txt line may name it) is written unless this arm qualifies for that run, or this node is the full-run node (klf) and a request is pending.
if [ -n "${POSTSTEP:-}" ] && has READ_s4400_full && has READ_prefix_full && ! has POST_DONE; then      # K24 (TA agent): the comparison with the soup and the full public run come BEFORE the DONE marker (a relaunch resumes them); the step's script is fetched fresh from S3 each time
  push_logs; aws s3 cp --only-show-errors $R/esbar_kl_code/scripts/$POSTSTEP $SC/$POSTSTEP.new && mv $SC/$POSTSTEP.new $SC/$POSTSTEP
  KL_ARM=$ARM KL_RUN=$RUN KL_PX=${PX:-} bash $SC/$POSTSTEP >> $F/logs/post.out 2>&1; rc=$?; log "POST $POSTSTEP rc=$rc: $(tail -1 $F/logs/post.log 2>/dev/null | cut -c1-300)"
  [ $rc = 0 ] || die "POST $POSTSTEP rc=$rc"; mark POST_DONE "$(tail -1 $F/logs/post.log 2>/dev/null | cut -c1-300)"
fi
has READ_s4400_full && has READ_prefix_full && { mark DONE; log "KL_RUN_DONE $ARM prefix ${PX:-?}"; push_logs; CAND=0
  case $ARM in k20|k21|k23) [ -n "${PX:-}" ] && python3 -c "import sys; sys.exit(0 if float('$PX') >= 73.95 else 1)" && CAND=1;; esac
  if [ -n "${SELFFULL:-}" ] && [ -n "${PX:-}" ] && python3 -c "import sys; sys.exit(0 if float('$PX') >= $SELFFULL else 1)"; then      # coordinator 19:25Z 11-16: K22 is a candidate; its full public run on THIS node right after its prefix read
    if aws s3 ls $R/esbar_kl_${ARM}_full/kown_full/KFULL.txt > /dev/null 2>&1; then log "SELFFULL: full public run exists already"
    else cut -c1-64 $F/logs/identity_export_full.txt > $RUN/export_full/.identity 2>/dev/null; log "SELFFULL: final prefix $PX >= $SELFFULL: FULL PUBLIC run on this node now (mx_kfull.sh kl_${ARM}_full; nothing public)"
      bash $M/scripts/mx_kfull.sh kl_${ARM}_full $RUN/export_full > $F/logs/selffull.out 2>&1; rc=$?
      log "SELFFULL: mx_kfull.sh rc=$rc: $(grep -E 'KFULL_DONE|DIE' $M/work/kl_${ARM}_full/kfull/driver.log 2>/dev/null | tail -1 | cut -c1-300)"
      [ $rc = 0 ] && echo "$(date -u +%FT%TZ) $(hostname) $(grep -E '^HEADLINE' $M/work/kl_${ARM}_full/kfull/merge/report.txt 2>/dev/null | cut -c1-300)" | aws s3 cp --only-show-errors - $O/markers/FULL_DONE; fi
  elif [ -n "${SELFFULL:-}" ]; then log "SELFFULL: final prefix ${PX:-?} < $SELFFULL: no full public run"; fi
  PEND=0; [ "$NODE" = klf ] && aws s3 ls $R/esbar_kl_code/full/ 2>/dev/null | awk '{print $4}' | grep -qx REQUEST && PEND=1
  if [ $CAND = 0 ] && [ $PEND = 0 ]; then echo "$(date -u +%FT%TZ) $(hostname) $ARM done on $NODE (final prefix ${PX:-?}; not a full-run candidate)" | aws s3 cp --only-show-errors - $R/esbar_kl_code/full/ARMDONE_$ARM; fi; }
push_logs
