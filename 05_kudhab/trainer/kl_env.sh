# kl_env.sh (sourced): the KL agent's node environment. venv_mxk = karpo's own run environment (torch 2.14.1+cu130, transformers 5.21.0).
unset ASCALE_RUN_ID
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_RUNTIME_CONNECT=0
D=/data/local; K=$D/kl; M=$D/mx; V=$D/venv_mxk; C=$D/esbar_code_v2e; KR=$M/kroot
S3=s3://YOUR_BUCKET/jev-race; E=$S3/engineering/esbar; R=$E/runs; SC=$K/scripts
export PYTHONPATH=$KR/karpo/src:$SC
MUNKU=$D/soup75_in/munku; KARPO=$M/src/karpo
TR="$V/bin/python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --local-addr 127.0.0.1"
klget() { mkdir -p $SC; aws s3 sync --only-show-errors $R/esbar_kl_code/scripts/ $SC/; }
