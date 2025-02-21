
# JSON PATH
DATA_PATH=./data/path
SAVE_PATH=./save/path
MODEL_CONFIG=./configs/inference/cldm.yaml

BSRNET_PATH=./path/ckpts/BSRNET.pth
MODEL_PATH=./path/ckpts/tasr_v1.pt

CUDA_VISIBLE_DEVICES=2 python -u /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/linqinwei/code/TASR/TASR/inference.py \
--version v2 \
--task sr \
--upscale 4 \
--cfg_scale 4 \
--pos_prompt ",high quality, clear text" \
--steps 20 \
--tag_prompt \
--input $JSON_DIV2K \
--output $SAVE_PATH \
--device cuda \
--bsrnet $BSRNET_PATH \
--control_adapter $MODEL_PATH \
--model_config $MODEL_CONFIG 