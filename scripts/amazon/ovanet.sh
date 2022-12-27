

export CUDA_VISIBLE_DEVICES=3


lrs='1e-5 5e-6 1e-6'


seeds='1234 2134 3412 4132'


# # B -> D, E, k

# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-books-dvd.yaml --lr $lr --seed $seed
#     done
# done


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-books-electronics.yaml --lr $lr --seed $seed
#     done
# done


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-books-kitchen.yaml --lr $lr --seed $seed
#     done
# done


# # D -> B, E, K

# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-dvd-books.yaml --lr $lr --seed $seed
#     done
# done


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-dvd-electronics.yaml --lr $lr --seed $seed
#     done
# done


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-dvd-kitchen.yaml --lr $lr --seed $seed
#     done
# done


# E -> B, D, K

# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-electronics-books.yaml --lr $lr --seed $seed
#     done
# done


# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/ovanet.py --config configs/nlp/ovanet-amazon-electronics-dvd.yaml --lr $lr --seed $seed
#     done
# done


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-electronics-kitchen.yaml --lr $lr --seed $seed
    done
done


# K -> B, D, E

# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-kitchen-books.yaml --lr $lr --seed $seed
    done
done


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-kitchen-dvd.yaml --lr $lr --seed $seed
    done
done


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-kitchen-electronics.yaml --lr $lr --seed $seed
    done
done