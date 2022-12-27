

export CUDA_VISIBLE_DEVICES=0


lrs='5e-5 1e-5 5e-6'

seeds='1234 2134 3412 4132'

# # adv. weight = 0.8
# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-clinc-oda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 2
#     done
# done


# # adv. weight = 0.5
# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-clinc-oda_0.5.yaml --lr $lr --seed $seed --num_nli_sample 2
#     done
# done



# # adv. weight = 0.1
# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-clinc-oda_0.1.yaml --lr $lr --seed $seed --num_nli_sample 2
#     done
# done

# adv. weight = 0.6
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-clinc-oda_0.6.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# adv. weight = 0.7
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-clinc-oda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done