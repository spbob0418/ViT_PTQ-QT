

DIR=/home/shkim/SSF_org/SSF
VERSION=full_8reg_qkcos_01_mean_5e-5_model_learnable
MODEL=fullbits_vit_base_patch16_224
export DIR VERSION MODEL
cd ${DIR}
mkdir -p ${DIR}/output/${MODEL}/cifar100/${VERSION}
MASTER_PORT=$((RANDOM % 10000 + 10000))

nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 taskset -c 32-63 python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} \
	train_with_newloss.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100 --model ${MODEL} \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${DIR}/output/${MODEL}/cifar100/${VERSION} \
	--register-num 8 \
	--save-custom-cp \
	--save-cp-point 0 1 2 3 4 5 10 --checkpoint-hist 8 \
	--prompt-search-mode-with-custom-loss \
	--loss-mode mean \
	--lambda-coef 0.1 \
	--pretrained" > ${DIR}/output/${MODEL}/cifar100/${VERSION}/train.log 2>&1 &

	# --log-wandb --experiment ${MODEL}-${VERSION} \
