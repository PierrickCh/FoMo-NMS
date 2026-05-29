#for split in roof_slate_512; do # plank_floor_512 wooden_floor_512 stone_tiles_512 ; do
for split in $(seq 1 15); do   
for lam_gan in 0.01; do
for lr in 1e-4; do
for bs in 8; do
for dim in 32; do
for training_steps in 20000; do
for img_size in 256; do
for fourier_mode in fomonms UFU_official FU Attention Vanilla; do
dataset_path=./images/data/${split}/
if [[ "$fourier_mode" == "fomonms" ]]; then
    nms_sizes=(3)
    EMN_list=(1 0)
    CA_list=(1 0)
else
    nms_sizes=(3)
    EMN_list=(0)
    CA_list=(0)
fi

for CA in "${CA_list[@]}"; do
for EMN in "${EMN_list[@]}"; do
for nms_size in "${nms_sizes[@]}"; do
    name="img_$(basename "$dataset_path")_${fourier_mode}"
    if [ "$fourier_mode" = "fomonms" ]; then
        name="${name}_size_${nms_size}_EMN_${EMN}_CA_${CA}"
    fi
    echo "$name"
    sbatch  job.sh \
        "$name" \
        "$dataset_path" \
        "$fourier_mode" \
        "$lam_gan" \
        "$lr" \
        "$training_steps" \
        "$dim" \
        "$bs" \
        "$img_size" \
        "$nms_size" \
        "$CA" \
        "$EMN"
done
done
done
done
done
done
done
done
done
done
done
