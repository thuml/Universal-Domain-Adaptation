

export CUDA_VISIBLE_DEVICES=0


# lrs='5e-5 1e-5 5e-6'
# seeds='1234 2134 3412 4132'

seeds='1234 2134'
lrs='5e-6'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-massive-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done

seeds='3412'
lrs='5e-5'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-massive-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done

seeds='4132'
lrs='1e-5'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-massive-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 4
    done
done