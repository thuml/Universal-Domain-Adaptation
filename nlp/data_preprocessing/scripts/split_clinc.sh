

seeds='1234 2134 3412 4132'
# seeds='1234'
# num_common_class='8'
# num_common_class='10'
num_common_classes='2 6 8'

for num_common_class in $num_common_classes; do
    for seed in $seeds; do
        python nlp/data_preprocessing/split_clinc.py \
            --seed $seed \
            --num_common_class $num_common_class
    done
done