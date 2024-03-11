#!/bin/bash
#SBATCH --job-name=rsync_pis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-3
#SBATCH --partition=cpu
#SBATCH --mem=10G
#SBATCH --time=08:00:00

# Input file of IPs
# Generated from the csv using awk -F "\"*,\"*" '{print $2}' inventory_for_rsync.csv > pi_ips.txt
# because it's a bit easier
ids_file="ip.txt"
ip_var=$( sed -n ${SLURM_ARRAY_TASK_ID}p $ids_file )

# rsync using the IP address obtained above

echo $ip_var

rsync -avzh --progress --remove-source-files plugcamera@$ip_var:/home/plugcamera/data/ /camp/lab/windingm/data/instruments/behavioural_rigs/plugcamera/data/2024-02-27_3hr-staging
ssh plugcamera@$ip_var "find data/ -mindepth 1 -type d -empty -delete"