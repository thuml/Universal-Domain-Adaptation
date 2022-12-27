

export CUDA_VISIBLE_DEVICES=1


lrs='5e-5 1e-5 5e-6'

seeds='1234 2134 3412 4132'

# # adv. weight = 0.0
# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/udanli_v9-1.py --config configs/nlp/udanli/udanli-clinc-opda_0.0.yaml --lr $lr --seed $seed --num_nli_sample 2
#     done
# done


# adv. weight = 0.1
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/udanli-clinc-opda_0.1.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


# adv. weight = 0.2
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/udanli-clinc-opda_0.2.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


# adv. weight = 0.3
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/udanli-clinc-opda_0.3.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


# adv. weight = 0.4
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/udanli-clinc-opda_0.6.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


# adv. weight = 0.9
# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/udanli-clinc-opda_0.9.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done