

export CUDA_VISIBLE_DEVICES=1



# fine-tuning
method='fine_tuning'
seed='1234'
lr='5e-5'
threshold='0.53'
python nlp/evaluation.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold