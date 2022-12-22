

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412 4132'


lrs='5e-5 1e-5 6e-6'

source='books'
target='kitchen'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/udalm-amazon-$source-$target.yaml --lr $lr --seed $seed --batch_size 8
    done
done

