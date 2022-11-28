export CUDA_VISIBLE_DEVICES=2


# lrs='1e-2 5e-3 1e-3 5e-4'
lrs='1e-3 5e-4 1e-2 5e-3'

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

lrs='5e-4 1e-2 5e-3'
# amazon -> dslr
for lr in $lrs; do
    python cmu.py --config configs/cmu-office-train-amazon-dslr.yaml --lr $lr
done


lrs='1e-3 5e-4 1e-2 5e-3'
# amazon -> webcam
for lr in $lrs; do
    python cmu.py --config configs/cmu-office-train-amazon-webcam.yaml --lr $lr
done

# # dslr -> amazon
# for lr in $lrs; do
#     python cmu.py --config configs/cmu-office-train-dslr-amazon.yaml --lr $lr
# done

# # dslr -> webcam
# for lr in $lrs; do
#     python cmu.py --config configs/cmu-office-train-dslr-webcam.yaml --lr $lr
# done
