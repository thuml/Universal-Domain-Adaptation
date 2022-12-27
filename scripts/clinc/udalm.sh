

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412 4132'


lrs='5e-5 1e-5 6e-6'


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udalm.py --config configs/nlp/udalm-clinc-opda.yaml --lr $lr --seed $seed
#     done
# done


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udalm.py --config configs/nlp/udalm-clinc-cda.yaml --lr $lr --seed $seed
#     done
# done


# ODA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm_oda.py --config configs/nlp/udalm-clinc-oda.yaml --lr $lr --seed $seed
    done
done