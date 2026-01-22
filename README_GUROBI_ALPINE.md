# Gurobi Setup on Alpine

## Get Gurobi Academic Lisence (WLS version)

## Install Gurobi on Alpine
Run:

```bash
cd /scratch/alpine/$USER
mkdir -p software
cd software

# Upload/download Gurobi from https://www.gurobi.com/downloads/
tar -xzf gurobi13.0.0_linux64.tar.gz
mv gurobi13.0.0_linux64 gurobi1300
```

Directory should look like:

```
/scratch/alpine/$USER/software/
 ├── gurobi1300/
 └── gurobi.lic
```

---

## Copy License File

Place your license at (path is optional, just for tideness):

```
/scratch/alpine/$USER/software/gurobi.lic
```

Verify:

```bash
ls -l /scratch/alpine/$USER/software/gurobi.lic
```

---

## Environment Variables

Add to `~/.bashrc` **and** include in SLURM scripts:

```bash
export GUROBI_HOME=/scratch/alpine/$USER/software/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=/scratch/alpine/$USER/software/gurobi.lic
```

Reload:

```bash
source ~/.bashrc
```

Test:

```bash
which gurobi_cl
echo $GRB_LICENSE_FILE
```

---


## Python Integration

```bash
pip install gurobipy==13.0.0
```

Test:

```python
import gurobipy as gp
m = gp.Model()
print("Gurobi Version:", gp.gurobi.version())
```

---

## SLURM Job Example (Works for Python + Julia)

```bash
#!/bin/bash
#SBATCH --job-name=gurobi-test
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/alpine/%u/logs/%x_%j.out

# === Gurobi ===
export GUROBI_HOME=/scratch/alpine/$USER/software/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=/scratch/alpine/$USER/software/gurobi.lic

python script.py
```