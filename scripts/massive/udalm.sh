

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412 4132'
seeds='3412'


lrs='5e-5 1e-5 5e-6'
# lrs='1e-3 5e-4 1e-4'

# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udalm.py --config configs/nlp/udalm-massive-opda.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udalm.py --config configs/nlp/udalm-massive-cda.yaml --lr $lr --seed $seed
#     done
# done


# ODA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm_oda.py --config configs/nlp/udalm-massive-oda.yaml --lr $lr --seed $seed
    done
done