

export CUDA_VISIBLE_DEVICES=3


lrs='1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'



# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-books-dvd.yaml --lr $lr --seed $seed
    done
done


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-books-electronics.yaml --lr $lr --seed $seed
    done
done


# OPDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-amazon-books-kitchen.yaml --lr $lr --seed $seed
    done
done