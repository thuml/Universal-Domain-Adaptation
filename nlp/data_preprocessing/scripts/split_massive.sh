

seeds='1234 2134 3412 4132'
seeds='1234 2134 3412'

num_common_classes='2 4 6 10 12 14 16'

for num_common_class in $num_common_classes; do
    for seed in $seeds; do
        python nlp/data_preprocessing/split_massive.py \
            --seed $seed \
            --num_common_class $num_common_class
    done
done