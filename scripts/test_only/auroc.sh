
export CUDA_VISIBLE_DEVICES=1

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam


method='uan'

# amazon -> dslr
lr='1e-3'
threshold='-0.5'
python auroc.py --config configs/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --min_threshold -1.0 --max_threshold 1.0

# amazon -> webcam
lr='5e-4'
threshold='-0.5'
python auroc.py --config configs/fine_tuning-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --min_threshold -1.0 --max_threshold 1.0
