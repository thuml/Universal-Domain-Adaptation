

CUDA_VISIBLE_DEVICES=0


lrs='1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7'


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# source : amazon (0)
# target : dslr  (1)

for lr in $lrs; do
    python uan.py --config configs/office-train-config.yaml --lr $lr
done