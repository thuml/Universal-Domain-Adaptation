

export CUDA_VISIBLE_DEVICES=2


lrs='1e-2 5e-3 1e-3 5e-4'


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# DONE
# amazon -> dslr
for lr in $lrs; do
    python uan.py --config configs/full_iter/uan-office-train-amazon-dslr.yaml --lr $lr
done

# amazon -> webcam
for lr in $lrs; do
    python uan.py --config configs/full_iter/uan-office-train-amazon-webcam.yaml --lr $lr
done

# dslr -> amazon
for lr in $lrs; do
    python uan.py --config configs/full_iter/uan-office-train-dslr-amazon.yaml --lr $lr
done

# dslr -> webcam
for lr in $lrs; do
    python uan.py --config configs/full_iter/uan-office-train-dslr-webcam.yaml --lr $lr
done
