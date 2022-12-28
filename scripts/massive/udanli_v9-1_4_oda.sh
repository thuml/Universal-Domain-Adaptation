

export CUDA_VISIBLE_DEVICES=2


lrs='5e-5 1e-5 5e-6'


seeds='1234 2134 3412 4132'
# seeds='1234 2134 4132'


# 0.7
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-massive-oda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done

# 0.8
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-massive-oda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done


# 0.5
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-massive-oda_0.5.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done


# 0.6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_oda.py --config configs/nlp/udanli/udanli-massive-oda_0.6.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done