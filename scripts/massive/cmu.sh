

export CUDA_VISIBLE_DEVICES=1


lrs='5e-4 1e-4 5e-5 1e-5'

seeds='1234 2134 3412 4132'



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/cmu-massive-opda.yaml --lr $lr --seed $seed
    done
done
