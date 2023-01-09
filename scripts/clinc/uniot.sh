

export CUDA_VISIBLE_DEVICES=0

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'
# lrs='1e-2 5e-3 1e-3 5e-4'
lrs='1e-3 5e-4 1e-4 5e-5'

seeds='1234 2134 3412 4132'


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uniot.py --config configs/nlp/uniot-clinc-opda.yaml --lr $lr --seed $seed
    done
done

sh scripts/massive/uniot.sh