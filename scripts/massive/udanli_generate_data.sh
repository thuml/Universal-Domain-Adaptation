

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412 4132'
# seeds='2134 3412 4132'

num_samples='4'

for seed in  $seeds; do
    for num_sample in $num_samples; do
        python nlp/udanli_generate_dataset.py --config configs/nlp/udanli-massive-opda_0.6.yaml --seed $seed --num_nli_sample $num_sample
    done
done
