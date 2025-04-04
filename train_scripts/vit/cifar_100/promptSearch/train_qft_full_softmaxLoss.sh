

DIR=/home/shkim/SSF_org/SSF
VERSION=test_for_softmax
MODEL=fullbits_vit_base_patch16_224
export DIR VERSION MODEL
mkdir -p ${DIR}/output/${MODEL}/cifar100/${VERSION}
MASTER_PORT=$((RANDOM % 10000 + 10000))

nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} \
	train_softmax_loss.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100 --model ${MODEL} \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr 4e-4 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${DIR}/output/${MODEL}/cifar100/${VERSION} \
	--prompt-search-mode \
	--register-num 8 \
	--log-wandb --experiment ${MODEL}-${VERSION} \
	--save-cp-point 0 1 2 3 4 5 10 --checkpoint-hist 8 \
	--pretrained" > ${DIR}/output/${MODEL}/cifar100/${VERSION}/train.log 2>&1 &