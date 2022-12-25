

export CUDA_VISIBLE_DEVICES=0


lrs='1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'




for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v10-3.py --config configs/nlp/udanli_v10/udanli-clinc-opda_0.3_0.05.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v10-3.py --config configs/nlp/udanli_v10/udanli-clinc-opda_0.3_0.1.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v10-3.py --config configs/nlp/udanli_v10/udanli-clinc-opda_0.3_0.2.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v10-3.py --config configs/nlp/udanli_v10/udanli-clinc-opda_0.3_0.3.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done