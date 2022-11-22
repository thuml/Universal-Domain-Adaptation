
export CUDA_VISIBLE_DEVICES=2


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam



method='fine_tuning'

# amazon -> dslr
lr='1e-2'
threshold='0.71'
python auroc.py --config configs/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0

# amazon -> webcam
lr='5e-3'
threshold='0.745'
python auroc.py --config configs/fine_tuning-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0
