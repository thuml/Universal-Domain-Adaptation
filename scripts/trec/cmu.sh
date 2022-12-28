

export CUDA_VISIBLE_DEVICES=0

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/cmu-trec-opda.yaml --lr $lr --seed $seed
    done
done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-trec-cda.yaml --lr $lr --seed $seed
#     done
# done


# # ODA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu_oda.py --config configs/nlp/cmu-trec-oda.yaml --lr $lr --seed $seed
#     done
# done

sh scripts/trec/ovanet.sh