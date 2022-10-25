CUDA_VISIBLE_DEVICES=$1 \
python ./baseline/train_S_model.py \
    --dataset $2 \
    --seed 11 \
    --lr 0.001 \
    --epochs 100 \
    --label_rate 0.01 \
    --ptb_rate 0.2 \
    --noise uniform

CUDA_VISIBLE_DEVICES=$1 \
python  ./baseline/train_S_model.py \
    --dataset $2 \
    --seed 11 \
    --lr 0.001 \
    --epochs 100 \
    --label_rate 0.01 \
    --ptb_rate 0.2 \
    --noise pair

# Coteaching
CUDA_VISIBLE_DEVICES=$1 \
python ./baseline/train_Coteaching.py \
    --dataset $2 \
    --seed 11 \
    --lr 0.001 \
    --epochs 100 \
    --label_rate 0.01 \
    --ptb_rate 0.2 \
    --noise uniform

CUDA_VISIBLE_DEVICES=$1 \
python  ./baseline/train_Coteaching.py \
    --dataset $2 \
    --seed 11 \
    --lr 0.001 \
    --epochs 100 \
    --label_rate 0.01 \
    --ptb_rate 0.2 \
    --noise pair


# label_rate: rate of labeled data
# t_mall: threshold of eliminating the edges
# ptb_rate: noise ptb rate
# LOSS Detail:
# alpha:  weight of loss of edge predictor
# beta:   weight of the loss on pseudo labels
# p_u:    threshold of adding pseudo labels
# n_p:    number of positive pairs when supervising
# n_n:    number of negitive pairs when supervising
