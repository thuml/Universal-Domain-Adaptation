

export CUDA_VISIBLE_DEVICES=1


lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'
seeds='1234'



# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/fine_tuning.py --config configs/nlp/fine_tuning-clinc-opda.yaml --lr $lr --seed $seed
    done
done

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/fine_tuning.py --config configs/nlp/fine_tuning-clinc-cda.yaml --lr $lr --seed $seed
#     done
# done

# # ODA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/fine_tuning_oda.py --config configs/nlp/fine_tuning-clinc-oda.yaml --lr $lr --seed $seed
#     done
# done


# sh scripts/clinc/dann.sh