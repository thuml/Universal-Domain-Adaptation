

export CUDA_VISIBLE_DEVICES=2


# uan
method='uan'
threshold='-0.5'

seed='1234'
lr='5e-6'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='2134'
lr='5e-6'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='3412'
lr='1e-5'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='4132'
lr='5e-5'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# cmu
method='cmu'
threshold='0.5'

seed='1234'
lr='1e-5'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='2134'
lr='5e-6'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='3412'
lr=' 5e-6'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='4132'
lr='1e-5'
python nlp/evaluation_oda_opt_msp.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold



