

seeds='1234 2134 3412 4132'
# seeds='2134 3412 4132'
num_common_class='14'
num_common_class='18'
# 4 8 12 14

for seed in $seeds; do
    python ./data_preprocessing/split_massive.py \
        --seed $seed \
        --num_common_class $num_common_class
done