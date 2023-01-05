

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412 4132'


lrs='5e-5 1e-5 6e-6'


# num common class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-clinc-2.yaml --lr $lr --seed $seed
    done
done

# num common class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-clinc-6.yaml --lr $lr --seed $seed
    done
done

# num common class = 8
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-clinc-8.yaml --lr $lr --seed $seed
    done
done
