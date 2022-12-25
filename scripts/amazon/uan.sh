

export CUDA_VISIBLE_DEVICES=1


lrs='1e-5 5e-6 1e-6 5e-7'

seeds='1234 2134 3412 4132'

# # B -> D, E, K
# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/uan.py --config configs/nlp/uan-amazon-books-dvd.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/uan.py --config configs/nlp/uan-amazon-books-electronics.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/uan.py --config configs/nlp/uan-amazon-books-kitchen.yaml --lr $lr --seed $seed
#     done
# done


# D -> B, E, K
# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-dvd-books.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-dvd-electronics.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-dvd-kitchen.yaml --lr $lr --seed $seed
    done
done


# E -> B, D, K
# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-electronics-books.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-electronics-dvd.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-electronics-kitchen.yaml --lr $lr --seed $seed
    done
done


# K -> B, D, E
# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-kitchen-books.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-kitchen-dvd.yaml --lr $lr --seed $seed
    done
done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/uan.py --config configs/nlp/uan-amazon-kitchen-electronics.yaml --lr $lr --seed $seed
    done
done