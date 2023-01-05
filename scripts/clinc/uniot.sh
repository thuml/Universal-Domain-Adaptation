

export CUDA_VISIBLE_DEVICES=2

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'


lrs='1e-4'
seeds='1234'

# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uniot.py --config configs/nlp/uniot-clinc-opda.yaml --lr $lr --seed $seed
    done
done


# # ODA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/uniot_oda.py --config configs/nlp/uniot-clinc-oda.yaml --lr $lr --seed $seed
#     done
# done
