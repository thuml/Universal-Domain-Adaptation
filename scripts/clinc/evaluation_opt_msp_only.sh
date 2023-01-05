

export CUDA_VISIBLE_DEVICES=2



lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412'


# uan
method='uan'

# Common Class = 2
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/evaluation_opt_msp_only.py --config configs/nlp/ablation/uan-clinc-2.yaml --method_name $method --lr $lr --seed $seed 
    done
done

# Common Class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/evaluation_opt_msp_only.py --config configs/nlp/ablation/uan-clinc-6.yaml --method_name $method --lr $lr --seed $seed 
    done
done

# Common Class = 8
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/evaluation_opt_msp_only.py --config configs/nlp/ablation/uan-clinc-8.yaml --method_name $method --lr $lr --seed $seed 
    done
done

