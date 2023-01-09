

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412'

num_samples='2'

# num common class = 2
for seed in  $seeds; do
    for num_sample in $num_samples; do
        python nlp/udanli_generate_dataset.py --config configs/nlp/udanli/ablation/clinc-opda-2.yaml --seed $seed --num_nli_sample $num_sample
    done
done

# num common class = 6
for seed in  $seeds; do
    for num_sample in $num_samples; do
        python nlp/udanli_generate_dataset.py --config configs/nlp/udanli/ablation/clinc-opda-6.yaml --seed $seed --num_nli_sample $num_sample
    done
done

# num common class = 8
for seed in  $seeds; do
    for num_sample in $num_samples; do
        python nlp/udanli_generate_dataset.py --config configs/nlp/udanli/ablation/clinc-opda-8.yaml --seed $seed --num_nli_sample $num_sample
    done
done
