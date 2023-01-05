
export CUDA_VISIBLE_DEVICES=2

################
#              #
#     1234     #
#              #
################

seed='1234'

###########################################################################
method='uniot'

# # amazon -> dslr
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-amazon-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01

# # dslr -> webcam
# lr='1e-2'
# python auroc.py --config configs/vision/uniot-office-train-dslr-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01

# # webcam -> dslr
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-webcam-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01


# ###########################################################################
method='fine_tuning'

# # amazon -> dslr
# lr='1e-2'
# threshold='0.71'
# python auroc.py --config configs/vision/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='5e-3'
# threshold='0.745'
# python auroc.py --config configs/vision/fine_tuning-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='1e-3'
# threshold='0.415'
# python auroc.py --config configs/vision/fine_tuning-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='5e-3'
# threshold='0.725'
# python auroc.py --config configs/vision/fine_tuning-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='1e-3'
# threshold='0.575'
# python auroc.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02


###########################################################################
method='uan'
threshold='-0.5'

# # amazon -> dslr
# lr='1e-3'
# python auroc.py --config configs/vision/uan-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='1e-3'
# python auroc.py --config configs/vision/uan-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> amazon
# lr='1e-3'
# python auroc.py --config configs/vision/uan-office-train-webcam-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02



# ###########################################################################
# method='ovanet'

# # amazon -> dslr
# lr='1e-2'
# python auroc.py --config configs/vision/ovanet-office-train-amazon-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='1e-4'
# python auroc.py --config configs/vision/ovanet-office-train-amazon-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='5e-3'
# python auroc.py --config configs/vision/ovanet-office-train-dslr-amazon.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='5e-3'
# python auroc.py --config configs/vision/ovanet-office-train-dslr-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='1e-3'
# python auroc.py --config configs/vision/ovanet-office-train-dslr-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02


###########################################################################
method='dann'

# # amazon -> dslr
# lr='1e-2'
# threshold='0.705'
# python auroc.py --config configs/vision/dann-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='1e-2'
# threshold='0.525'
# python auroc.py --config configs/vision/dann-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='1e-2'
# threshold='0.39'
# python auroc.py --config configs/vision/dann-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='1e-3'
# threshold='0.325'
# python auroc.py --config configs/vision/dann-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02


# ###########################################################################
method='cmu'
threshold='0.5'

# # amazon -> dslr
# lr='1e-3'
# threshold='0.5'
# python auroc.py --config configs/vision/cmu-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='1e-3'
# threshold='0.5'
# python auroc.py --config configs/vision/cmu-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='1e-2'
# threshold='0.5'
# python auroc.py --config configs/vision/cmu-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='1e-3'
# threshold='0.5'
# python auroc.py --config configs/vision/cmu-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02


################
#              #
#     2134     #
#              #
################

seed='2134'


###########################################################################
method='uniot'

# # amazon -> dslr
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-amazon-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01

# # dslr -> webcam
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-dslr-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01

# # webcam -> dslr
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-webcam-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01



##########################################################################

# method='fine_tuning'

# # webcam -> dslr
# lr='1e-3'
# threshold='0.675'
# python auroc.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02


# ###########################################################################
# method='ovanet'

# # amazon -> dslr
# lr='5e-4'
# python auroc.py --config configs/vision/ovanet-office-train-amazon-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='5e-4'
# python auroc.py --config configs/vision/ovanet-office-train-amazon-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='1e-2'
# python auroc.py --config configs/vision/ovanet-office-train-dslr-amazon.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='1e-2'
# python auroc.py --config configs/vision/ovanet-office-train-dslr-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> amazon
# lr='1e-2'
# python auroc.py --config configs/vision/ovanet-office-train-webcam-amazon.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='5e-3'
# python auroc.py --config configs/vision/ovanet-office-train-webcam-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# ###########################################################################
# method='cmu'
# threshold='0.5'

# # amazon -> dslr
# lr='5e-4'
# python auroc.py --config configs/vision/cmu-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='1e-3'
# python auroc.py --config configs/vision/cmu-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='1e-2'
# python auroc.py --config configs/vision/cmu-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # webcam -> amazon
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-webcam-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02


###########################################################################
method='dann'

# dslr -> webcam
lr='1e-3'
threshold='0.455'
python auroc.py --config configs/vision/dann-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02


################
#              #
#     3412     #
#              #
################

seed='3412'

###########################################################################
method='uniot'

# # amazon -> dslr
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-amazon-dslr.yaml --lr $lr --method $methodd --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01

# # dslr -> webcam
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-dslr-webcam.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01

# # webcam -> dslr
# lr='5e-2'
# python auroc.py --config configs/vision/uniot-office-train-webcam-dslr.yaml --lr $lr --method $method --seed $seed --min_threshold 0 --max_threshold 1.0 --step 0.01


###########################################################################
method='cmu'
threshold='0.5'

# # amazon -> dslr
# lr='5e-4'
# python auroc.py --config configs/vision/cmu-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='1e-3'
# python auroc.py --config configs/vision/cmu-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # webcam -> amazon
# lr='1e-3'
# python auroc.py --config configs/vision/cmu-office-train-webcam-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='5e-3'
# python auroc.py --config configs/vision/cmu-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold 0.0 --max_threshold 1.0 --step 0.02


###########################################################################
method='uan'
threshold='-0.5'

# # amazon -> dslr
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # amazon -> webcam
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-amazon-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> amazon
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-dslr-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> amazon
# lr='5e-4'
# python auroc.py --config configs/vision/uan-office-train-webcam-amazon.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='1e-3'
# python auroc.py --config configs/vision/uan-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02


# ###########################################################################
method='fine_tuning'

# # amazon -> dslr
# lr='5e-3'
# threshold='0.845'
# python auroc.py --config configs/vision/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # dslr -> webcam
# lr='1e-2'
# threshold='0.66'
# python auroc.py --config configs/vision/fine_tuning-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='1e-3'
# threshold='0.415'
# python auroc.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

# # webcam -> dslr
# lr='1e-2'
# threshold='0.835'
# python auroc.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02

###########################################################################
method='dann'

# amazon -> dslr
lr='5e-3'
threshold='0.59'
python auroc.py --config configs/vision/dann-office-train-amazon-dslr.yaml --lr $lr --method $method --threshold $threshold --seed $seed --min_threshold -1.0 --max_threshold 1.0 --step 0.02
