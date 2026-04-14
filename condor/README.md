# HTCondor GPU training on lxplus

## Files
- `condor/train_gpu.sub`: matrix-driven GPU submit description.
- `condor/jobs.tsv`: one row per training job (`label model dataset config extra_args`).
- `condor/run_train.sh`: per-job wrapper (LCG setup + validation + training call).
- `condor/submit_all.sh`: convenience submit helper.
- `scripts/setup_lcg.sh`: reusable LCG bootstrap script.

## Prepare environment
```bash
cd /path/to/MergedEleID_pytorch
source scripts/setup_lcg.sh
```

Optional layered venv on top of LCG Python:
```bash
export LCG_LAYERED_VENV=1
source scripts/setup_lcg.sh
```

## Local CPU dry run from lxplus login node
```bash
python train_zarr_weighted.py \
  --site lxplus \
  --dataset 260403_v1 \
  --label dryrun \
  --model swin_base_in22k \
  --config configs/default_train.json
```

## Submit full matrix
```bash
mkdir -p condor/logs
CPUS=4 MEMORY=16GB FLAVOUR=nextweek bash condor/submit_all.sh
```

Equivalent direct submission:
```bash
condor_submit -append "CPUS=4" -append "MEMORY=16GB" -append "FLAVOUR=nextweek" condor/train_gpu.sub
```

## Monitor jobs
```bash
condor_q $USER
condor_history $USER -limit 20
ls -ltr condor/logs
```

## Output location on EOS
Each run writes to:
`/eos/home-y/yeo/4l/image/<label>/<model>/<dataset>/<run_id>/`

`run_id` contains UTC timestamp + git SHA + Condor Cluster/Proc IDs when available.

Artifacts include checkpoints, plots, predictions, config snapshot, and `run.json` metadata.

## Extend the sweep
1. Add another line in `condor/jobs.tsv` with `label model dataset config extra_args`.
2. Re-submit using `condor/submit_all.sh`.
3. New models can be introduced by extending `MODEL_REGISTRY` in `train_zarr_weighted.py`.
