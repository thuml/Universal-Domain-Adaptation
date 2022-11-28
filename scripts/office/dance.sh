export CUDA_VISIBLE_DEVICES=2


lrs='1e-2 5e-3 1e-3 5e-4'
# lrs='1e-5 5e-5 1e-6 5e-6'
lrs='1e-2'

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# amazon -> dslr
for lr in $lrs; do
    python dance.py --config configs/dance-office-train-amazon-dslr.yaml --lr $lr
done

# # amazon -> webcam
# for lr in $lrs; do
#     python cmu.py --config configs/cmu-office-train-amazon-webcam.yaml --lr $lr
# done

# # dslr -> amazon
# for lr in $lrs; do
#     python cmu.py --config configs/cmu-office-train-dslr-amazon.yaml --lr $lr
# done

# # dslr -> webcam
# for lr in $lrs; do
#     python cmu.py --config configs/cmu-office-train-dslr-webcam.yaml --lr $lr
# done
