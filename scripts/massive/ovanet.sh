

export CUDA_VISIBLE_DEVICES=2

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'

# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-massive-opda.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-massive-cda.yaml --lr $lr --seed $seed
#     done
# done



lrs='1e-6'

seeds='3412'
# ODA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet_oda.py --config configs/nlp/ovanet-massive-oda.yaml --lr $lr --seed $seed
    done
done
