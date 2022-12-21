

export CUDA_VISIBLE_DEVICES=3


lrs='5e-6'
seeds='1234 2134 4132'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-clinc-opda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

lrs='1e-5'
seeds='3412'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-clinc-opda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done
