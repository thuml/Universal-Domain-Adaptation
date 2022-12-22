

export CUDA_VISIBLE_DEVICES=1


lrs='1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'

# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-books-dvd.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-books-electronics.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-books-kitchen.yaml --lr $lr --seed $seed
    done
done