
CUDA_VISIBLE_DEVICES=5 \
python train.py \
--job_dir results/1029_ori_len200_bs1024_inputxy_lr5e5

CUDA_VISIBLE_DEVICES=3 \
python train.py \
--job_dir results/1029_ori_len200_bs1024_inputxy_withcond_lossrecon_lr2e4






CUDA_VISIBLE_DEVICES=6 \
python evaluate.py 