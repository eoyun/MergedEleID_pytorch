# MergedEleID PyTorch training

## lxplus + HTCondor workflow

### 1) Setup repository on lxplus
```bash
git clone <your_repo_url>
cd MergedEleID_pytorch
source scripts/setup_lcg.sh
```

### 2) Test one training command on CPU (login node sanity check)
```bash
python train_zarr_weighted.py \
  --site lxplus \
  --dataset 260403_v1 \
  --label sanity \
  --model swin_base_in22k \
  --config configs/default_train.json
```

### 3) Build or edit the job matrix
Edit `condor/jobs.tsv` (columns: `label model dataset config extra_args`).

Example rows already include:
- datasets: `260403_v1`, `260405_v1`
- multiple models in parallel
- label: `baseline`

### 4) Submit many GPU jobs in parallel
```bash
CPUS=4 MEMORY=16GB FLAVOUR=nextweek bash condor/submit_all.sh
```

### 5) Inspect queue and logs
```bash
condor_q $USER
ls -ltr condor/logs
```

### 6) Output location on EOS
Persistent outputs are written under:
`/eos/home-y/yeo/4l/image/<label>/<model>/<dataset>/<run_id>/`

The run directory stores model checkpoints, metrics arrays, plots, predictions, run config, and `run.json` metadata.

### 7) Add new models
- Add model rows to `condor/jobs.tsv`.
- If using a new shorthand key, register it in `MODEL_REGISTRY` inside `train_zarr_weighted.py`.

See also: `condor/README.md`.
