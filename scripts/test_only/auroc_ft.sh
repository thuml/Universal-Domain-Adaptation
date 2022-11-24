
export CUDA_VISIBLE_DEVICES=2


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam



method='fine_tuning'

# amazon -> dslr
lr='1e-3'
threshold='0.415'
python auroc.py --config configs/fine_tuning-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0
