

export CUDA_VISIBLE_DEVICES=0



# # fine-tuning
# method='fine_tuning'

# seed='1234'
# lr='1e-5'
# threshold='0.95'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-4'
# threshold='0.925'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-4'
# threshold='0.625'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='0.695'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # dann
# method='dann'

# seed='1234'
# lr='1e-5'
# threshold='0.56'
# python nlp/evaluation_oda.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-4'
# threshold='0.48'
# python nlp/evaluation_oda.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-5'
# threshold='0.29'
# python nlp/evaluation_oda.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-4'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # uan
# method='uan'

# seed='1234'
# lr='5e-6'
# threshold='-0.4'
# python nlp/evaluation_oda.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-6'
# threshold='-0.22'
# python nlp/evaluation_oda.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-5'
# threshold='-0.35'
# python nlp/evaluation_oda.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='-0.028'
# python nlp/evaluation_oda.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # cmu
# method='cmu'

# seed='1234'
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-6'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr=' 5e-6'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# ovanet
method='ovanet'

seed='1234'
lr='1e-6'
python nlp/evaluation_oda.py --config configs/nlp/ovanet-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='1e-6'
# python nlp/evaluation_oda.py --config configs/nlp/ovanet-clinc-oda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='1e-6'
# python nlp/evaluation_oda.py --config configs/nlp/ovanet-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='1e-6'
# python nlp/evaluation_oda.py --config configs/nlp/ovanet-clinc-oda.yaml --method_name $method --lr $lr --seed $seed



# # udalm
# method='udalm'

# seed='1234'
# lr='5e-6'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='5e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='1e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='5e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed

