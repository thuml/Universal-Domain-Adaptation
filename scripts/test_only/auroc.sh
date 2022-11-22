

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
lrs='1e-2 5e-3 1e-3 5e-4'
lrs='1e-2'
for lr in $lrs; do
    python auroc.py --config configs/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method 'fine_tuning' --min_threshold 0.0 --max_threshold 1.0
done