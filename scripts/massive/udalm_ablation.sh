

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412'


lrs='5e-5 1e-5 5e-6'
# lrs='1e-3 5e-4 1e-4'

# num common class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-2.yaml --lr $lr --seed $seed
    done
done

# num common class = 4
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-4.yaml --lr $lr --seed $seed
    done
done

# num common class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-6.yaml --lr $lr --seed $seed
    done
done

# num common class = 10
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-10.yaml --lr $lr --seed $seed
    done
done

# num common class = 12
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-12.yaml --lr $lr --seed $seed
    done
done

# num common class = 14
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-14.yaml --lr $lr --seed $seed
    done
done

# num common class = 16
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/ablation/udalm-massive-16.yaml --lr $lr --seed $seed
    done
done
