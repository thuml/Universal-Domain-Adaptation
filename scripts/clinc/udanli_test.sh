

export CUDA_VISIBLE_DEVICES=1


lrs='1e-5'
seeds='1234 4132'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli/udanli-clinc-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

lrs='5e-5'
seeds='3412 4132'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli/udanli-clinc-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

lrs='5e-6'
seeds='2134'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_inference_only.py --config configs/nlp/udanli/udanli-clinc-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done