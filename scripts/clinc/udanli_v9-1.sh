

export CUDA_VISIBLE_DEVICES=3


lrs='5e-5 1e-5 5e-6'


seeds='1234 2134 3412'

# num common class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/ablation/clinc-opda-2.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/ablation/clinc-opda-6.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 8
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1.py --config configs/nlp/udanli/ablation/clinc-opda-2.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done
