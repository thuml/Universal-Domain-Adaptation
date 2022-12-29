

export CUDA_VISIBLE_DEVICES=1

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'
# lrs='5e-4 1e-3 5e-3'

seeds='1234 2134 3412 4132'
seeds='1234'

# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-trec-opda.yaml --lr $lr --seed $seed
    done
done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/uan.py --config configs/nlp/uan-trec-cda.yaml --lr $lr --seed $seed
#     done
# done


# # ODA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/uan_oda.py --config configs/nlp/uan-trec-oda.yaml --lr $lr --seed $seed
#     done
# done

# sh scripts/trec/cmu.sh