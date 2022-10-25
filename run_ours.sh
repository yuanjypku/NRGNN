Cpython train_ours.py \
    --dataset $2 \
    --seed 11 \
    --lr 0.001 \
    --epochs 100 \
    --label_rate 0.10 \
    --t_small 0.1 \
    --ptb_rate 0.2 \
    --noise uniform \
    --alpha 0.03\
    --p_u 0.8 \
    --n_p -1

CUDA_VISIBLE_DEVICES=$1 \
python train_ours.py \
    --dataset $2 \
    --seed 11 \
    --lr 0.001 \
    --epochs 100 \
    --label_rate 0.10 \
    --t_small 0.1 \
    --ptb_rate 0.2 \
    --noise pair \
    --alpha 0.03\
    --p_u 0.8 \
    --n_p -1
# label_rate: rate of labeled data
# t_mall: threshold of eliminating the edges
# ptb_rate: noise ptb rate
# LOSS Detail:
# alpha:  weight of loss of edge predictor
# beta:   weight of the loss on pseudo labels
# p_u:    threshold of adding pseudo labels
# n_p:    number of positive pairs when supervising
# n_n:    number of negitive pairs when supervising
