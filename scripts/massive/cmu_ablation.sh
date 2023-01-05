

export CUDA_VISIBLE_DEVICES=1

lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412'


# Common Class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-2.yaml --lr $lr --seed $seed
    done
done

# Common Class = 4
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-4.yaml --lr $lr --seed $seed
    done
done

# Common Class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-6.yaml --lr $lr --seed $seed
    done
done

# Common Class = 10
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-10.yaml --lr $lr --seed $seed
    done
done

# Common Class = 12
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-12.yaml --lr $lr --seed $seed
    done
done

# Common Class = 14
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-14.yaml --lr $lr --seed $seed
    done
done

# Common Class = 16
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/ablation/cmu-massive-16.yaml --lr $lr --seed $seed
    done
done

