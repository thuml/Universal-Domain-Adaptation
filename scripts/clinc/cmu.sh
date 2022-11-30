

export CUDA_VISIBLE_DEVICES=1


lrs='1e-4 5e-5 1e-5 5e-6 1e-5'

seeds='2134 3412 4132'



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/cmu-clinc-opda.yaml --lr $lr --seed $seed
    done
done
