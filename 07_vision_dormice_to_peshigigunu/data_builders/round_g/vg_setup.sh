#!/bin/bash
# round G: restore the round E / F code paths on the head's local disk (they are wiped when the head is replaced); idempotent
G=/data/shared/visdata_g; LV=/data/local/visdata
mkdir -p $LV/scripts $LV/forkb/scripts $LV/kit $G/scripts $G/logs
for f in vb_build.py vb_manifest.py; do [ -f $LV/scripts/$f ] || cp $G/base/$f $LV/scripts/$f; done
for f in ve_build.py ve_build2.py ve_build3.py ve_mix.py; do [ -f $LV/forkb/scripts/$f ] || cp $G/base/$f $LV/forkb/scripts/$f; done
[ -d $LV/kit/kit_vis ] || tar xzf $G/base/kit_vis.tgz -C $LV/kit
[ -f $LV/public_hashes.jsonl ] || cp $G/base/public_hashes.jsonl $LV/public_hashes.jsonl
for f in vf_build.py vf_build2.py vf_build3.py vf_build4.py vf_mix.py; do [ -f $G/scripts/$f ] || cp /data/shared/visdata_f/scripts/$f $G/scripts/$f; done
[ "$(md5sum < $LV/public_hashes.jsonl | cut -d' ' -f1)" = "ac38074e56fee2dcd685abc84f8a1dfe" ] || echo "SETUP_BAD public_hashes"
