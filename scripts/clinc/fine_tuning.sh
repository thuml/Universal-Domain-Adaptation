

export CUDA_VISIBLE_DEVICES=0


lrs='5e-4 1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'

# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/fine_tuning.py --config configs/nlp/fine_tuning-clinc-opda.yaml --lr $lr --seed $seed
#     done
# done

# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/fine_tuning.py --config configs/nlp/fine_tuning-clinc-cda.yaml --lr $lr --seed $seed
    done
done


sh scripts/massive/fine_tuning.sh