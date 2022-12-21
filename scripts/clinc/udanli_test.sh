

export CUDA_VISIBLE_DEVICES=3


lrs='1e-5'
seeds='2134 4132'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-clinc-opda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

lrs='5e-5'
seeds='1234 3412'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli-clinc-opda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done
