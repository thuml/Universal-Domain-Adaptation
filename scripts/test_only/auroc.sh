

CUDA_VISIBLE_DEVICES=1


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# # DONE
# # amazon -> dslr
# for lr in $lrs; do
#     python uan.py --config configs/office-train-amazon-dslr.yaml --lr $lr
# done

# amazon -> dslr
# lrs='1e-2'
# method='fine_tuning'
# threshold=''
# for lr in $lrs; do
#     python auroc.py --config configs/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method $method --min_threshold 0.0 --max_threshold 1.0
# done

lrs='5e-3'
threshold='0.51'


lrs='1e-2'
threshold='0.71'
method='fine_tuning'
for lr in $lrs; do
    python auroc.py --config configs/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0.0 --max_threshold 1.0
done