

export CUDA_VISIBLE_DEVICES=1

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412'

# common class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-2.yaml --lr $lr --seed $seed
    done
done

# common class = 4
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-4.yaml --lr $lr --seed $seed
    done
done

# common class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-6.yaml --lr $lr --seed $seed
    done
done

# common class = 10
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-10.yaml --lr $lr --seed $seed
    done
done

# common class = 12
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-12.yaml --lr $lr --seed $seed
    done
done

# common class = 14
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-14.yaml --lr $lr --seed $seed
    done
done

# common class = 16
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ablation/ovanet-massive-16.yaml --lr $lr --seed $seed
    done
done
