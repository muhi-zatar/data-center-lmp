# Running `data-center-lmp` on Alpine

This document describes the exact setup that worked for me to runn `data-center-lmp` on Alpine.

---

## 1. Login to Alpine

```bash
ssh <identikey>@login-ci4.rc.colorado.edu
```

---

## 2. Directory layout

Everything runs from **scratch** directory, not `$HOME`.

```
/scratch/alpine/$USER/
├── data-center-lmp/     # repository
├── venv/                # Python 3.11 virtual environment
├── logs/                # SLURM output logs
```

Verify the working directory:
```bash
pwd
# /scratch/alpine/muza5897/data-center-lmp
```

---

## 3. Python version

Default `python3` on Alpine is **3.6** which does not work for what we need.

Please check available system Python (anything 3.9 and above should work):
```
/usr/bin/python3.11
```

---

## 4. Create virtual environment with Python 3.11

```bash
cd /scratch/alpine/$USER

rm -rf venv
/usr/bin/python3.11 -m venv venv
source venv/bin/activate

python --version
# Python 3.11.x

pip install --upgrade pip
pip install -r /scratch/alpine/$USER/data-center-lmp/requirements.txt
```

---

## 5. Use absolute paths for scenario configs

Generate `configs.txt` with absolute paths:
Dont know why relative paths did not work.

Here you can create as many scenario configs as you need to run them in parallel (as parallel jobs)

```bash
REPO=/scratch/alpine/$USER/data-center-lmp
ls $REPO/configs/scenario*.yaml | sort > $REPO/configs.txt
```

Verify:
```bash
head configs.txt
```

Example entry:
```
/scratch/alpine/muza5897/data-center-lmp/configs/scenario_01.yaml
```

---

## 6. SLURM job array script (working)

File: `slurm/run_array.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=rt-sim
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/alpine/%u/logs/%x_%A_%a.out

set -euo pipefail

REPO=/scratch/alpine/$USER/data-center-lmp

mkdir -p /scratch/alpine/$USER/logs

source /scratch/alpine/$USER/venv/bin/activate
cd $REPO/src

CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $REPO/configs.txt)

echo "[${date}] Task=$SLURM_ARRAY_TASK_ID"
echo "Config=$CONFIG"

python run_scenario.py "$CONFIG" config.yaml
```

---

## 7. Submit jobs

```bash
cd /scratch/alpine/$USER/data-center-lmp
sbatch slurm/run_array.sbatch
```

---

## 8. Monitor jobs and logs

```bash
squeue -u $USER
tail -f /scratch/alpine/$USER/logs/rt-sim_<JOBID>_1.out
```

The command `squeue` tells you how long it has been running, it 5 minutes passed, it should continue fine.