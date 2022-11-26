
export CUDA_VISIBLE_DEVICES=2


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam



method='dann'



lr='1e-2'
threshold='0.705'
python auroc.py --config configs/dann-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0


lr='1e-2'
threshold='0.525'
python auroc.py --config configs/dann-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0
