

export CUDA_VISIBLE_DEVICES=0


lrs='1e-2 5e-3 1e-3'
# lrs='5e-3 1e-4 5e-5'


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

seeds='2134 3412 4132'

for seed in  $seeds; do
    lrs='1e-2'
    # amazon -> dslr
    for lr in $lrs; do
        python ovanet.py --config configs/ovanet-office-train-amazon-dslr.yaml --lr $lr --seed $seed
    done

    lrs='1e-2 1e-4'
    # amazon -> webcam
    for lr in $lrs; do
        python ovanet.py --config configs/ovanet-office-train-amazon-webcam.yaml --lr $lr --seed $seed
    done

    lrs='1e-2'
    # # dslr -> amazon
    for lr in $lrs; do
        python ovanet.py --config configs/ovanet-office-train-dslr-amazon.yaml --lr $lr --seed $seed
    done

    lrs='1e-2 1e-3'
    # dslr -> webcam
    for lr in $lrs; do
        python ovanet.py --config configs/ovanet-office-train-dslr-webcam.yaml --lr $lr --seed $seed
    done
done
