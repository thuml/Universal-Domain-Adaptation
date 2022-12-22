

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412 4132'


lr='1e-4'

source='books'
target='dvd'

for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/udalm-amazon-$source-$target.yaml --lr $lr --seed $seed --batch_size 8
done

