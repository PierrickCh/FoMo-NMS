#!/bin/bash
## Job name
#SBATCH -J "FoMo"

# Batch output file
#SBATCH --output /home/2024032/pchati01/logs/out/%j.out

# Batch error file
#SBATCH --error /home/2024032/pchati01/logs/err/%j.err

# Partition (submission class)
#SBATCH --partition gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1

# Job time (hh:mm:ss)
#SBATCH --time 5:00:00

# To stop the program if the validation is too long
#SBATCH --signal=SIGUSR2@60

##SBATCH  --array=1-10%1 
#SBATCH --mail-type=ARRAY_TASKS


# ------------------------------
module purge
module load aidl/pytorch/2.5.1-cuda12.4
PATH=$PATH:~/.python3-3.10-torch111/site_packages/bin
export PATH
# ------------------------------


cd ~/projects/FoMo-NMS/
python train.py --name $1 --dataset_path $2 --fourier_mode $3 --lam_gan $4 --lr $5 --training_steps $6 --dim $7 --bs $8 --img_size $9 --nms_size ${10} --save_every 5000 --CA ${11:-1} --EMN ${12:-1}
