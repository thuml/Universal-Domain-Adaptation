

export CUDA_VISIBLE_DEVICES=1


lrs='1e-2 5e-3 1e-3'
# lrs='5e-3 1e-4 5e-5'


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam




# dslr -> webcam
for lr in $lrs; do
    python ovanet_v2.py --config configs/ovanet-office-train-dslr-webcam.yaml --lr $lr
done

# amazon -> webcam
for lr in $lrs; do
    python ovanet_v2.py --config configs/ovanet-office-train-amazon-webcam.yaml --lr $lr
done

# # amazon -> dslr
for lr in $lrs; do
    python ovanet_v2.py --config configs/ovanet-office-train-amazon-dslr.yaml --lr $lr
done

# # dslr -> amazon
for lr in $lrs; do
    python ovanet_v2.py --config configs/ovanet-office-train-dslr-amazon.yaml --lr $lr
done