

export CUDA_VISIBLE_DEVICES=3


lrs='5e-5 1e-5 5e-6'


seeds='1234 2134 3412'

# num common class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-2.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 4
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-4.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-6.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 10
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-10.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 12
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-12.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 14
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-14.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

# num common class = 16
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9.py --config configs/nlp/udanli/ablation/massive-opda-16.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


sh scripts/clinc/udanli_v9-1.sh