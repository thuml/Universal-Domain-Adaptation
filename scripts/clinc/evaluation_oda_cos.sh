

export CUDA_VISIBLE_DEVICES=1



# fine-tuning
method='fine_tuning'

seed='1234'
lr='1e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='1e-4'
python nlp/evaluation_oda_cos.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='3412'
lr='1e-4'
python nlp/evaluation_oda_cos.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='5e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/fine_tuning-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 




# dann
method='dann'

seed='1234'
lr='1e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='1e-4'
python nlp/evaluation_oda_cos.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='3412'
lr='1e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='1e-4'
python nlp/evaluation_oda_cos.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 


# udalm
method='udalm'

seed='1234'
lr='5e-6'
python nlp/evaluation_oda_cos.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='5e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed

seed='3412'
lr='1e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='5e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/udalm-clinc-oda.yaml --method_name $method --lr $lr --seed $seed



# uan
method='uan'

seed='1234'
lr='5e-6'
python nlp/evaluation_oda_cos.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='5e-6'
python nlp/evaluation_oda_cos.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='3412'
lr='1e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='5e-5'
python nlp/evaluation_oda_cos.py --config configs/nlp/uan-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 




# cmu
method='cmu'

seed='1234'
lr='1e-5'
threshold='0.5'
python nlp/evaluation_oda_cos.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='5e-6'
threshold='0.5'
python nlp/evaluation_oda_cos.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='3412'
lr='5e-6'
threshold='0.5'
python nlp/evaluation_oda_cos.py --config configs/nlp/cmu-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='1e-5'
threshold='0.5'
python nlp/evaluation_oda_cos.py --config configs/nlp/dann-clinc-oda.yaml --method_name $method --lr $lr --seed $seed 




