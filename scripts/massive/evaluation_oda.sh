

export CUDA_VISIBLE_DEVICES=0



# # fine-tuning
# method='fine_tuning'

# seed='1234'
# lr='1e-5'
# threshold='0.62'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-6'
# threshold='0.755'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='5e-6'
# threshold='0.73'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-4'
# threshold='0.52'
# python nlp/evaluation_oda.py --config configs/nlp/fine_tuning-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # dann
# method='dann'

# seed='1234'
# lr='1e-4'
# threshold='0.18'
# python nlp/evaluation_oda.py --config configs/nlp/dann-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-6'
# threshold='0.18'
# python nlp/evaluation_oda.py --config configs/nlp/dann-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-5'
# threshold='0.16'
# python nlp/evaluation_oda.py --config configs/nlp/dann-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-4'
# threshold='0.81'
# python nlp/evaluation_oda.py --config configs/nlp/dann-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # uan
# method='uan'

# seed='1234'
# lr='1e-5'
# threshold='-0.37'
# python nlp/evaluation_oda.py --config configs/nlp/uan-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-5'
# threshold='0.2'
# python nlp/evaluation_oda.py --config configs/nlp/uan-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='5e-6'
# threshold='-0.42'
# python nlp/evaluation_oda.py --config configs/nlp/uan-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-4'
# threshold='0.44'
# python nlp/evaluation_oda.py --config configs/nlp/uan-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # cmu
# method='cmu'

# seed='1234'
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/cmu-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-5'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/cmu-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr=' 1e-4'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/cmu-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-6'
# threshold='0.5'
# python nlp/evaluation_oda.py --config configs/nlp/dann-massive-oda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# ovanet
method='ovanet'

# seed='1234'
# lr='1e-6'
# python nlp/evaluation_oda.py --config configs/nlp/ovanet-massive-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='1e-6'
# python nlp/evaluation_oda.py --config configs/nlp/ovanet-massive-oda.yaml --method_name $method --lr $lr --seed $seed

seed='3412'
lr='1e-6'
python nlp/evaluation_oda.py --config configs/nlp/ovanet-massive-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='1e-6'
# python nlp/evaluation_oda.py --config configs/nlp/ovanet-massive-oda.yaml --method_name $method --lr $lr --seed $seed



# # udalm
# method='udalm'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-massive-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='5e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-massive-oda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-massive-oda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='5e-5'
# python nlp/evaluation_oda.py --config configs/nlp/udalm-massive-oda.yaml --method_name $method --lr $lr --seed $seed

