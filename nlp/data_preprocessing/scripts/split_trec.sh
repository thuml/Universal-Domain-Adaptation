

seeds='1234 2134 3412 4132'
seeds='1234'
num_common_class='6'
num_common_class='2'

for seed in $seeds; do
    python nlp/data_preprocessing/split_trec.py \
        --seed $seed \
        --num_common_class $num_common_class
done