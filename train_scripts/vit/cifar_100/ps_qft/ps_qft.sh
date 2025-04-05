

DIR=/home/shkim/SSF_org/SSF
VERSION=8reg_ps4e-4_20_qft5e-5
MODEL=ps_qft

export DIR VERSION MODEL
mkdir -p ${DIR}/output/${MODEL}/cifar100/${VERSION}
MASTER_PORT=$((RANDOM % 10000 + 10000))

nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} \
	train.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100  \
    --batch-size 32 --epochs 100 \
	--opt adamw --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
	--min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${DIR}/output/${MODEL}/cifar100/${VERSION} \
	--save-best-cp --checkpoint-hist 2 \
	--log-wandb --experiment ${VERSION} \
	--register-num 8 \
	--ps-model fullbits_vit_base_patch16_224 \
	--prompt-searching-lr 4e-4 \
	--qft-model eightbits_vit_base_patch16_224 \
	--qft-lr 5e-5 \
	--prompt-search-epoch 20 \
	--a-quant-type per_tensor \
	--ag-quant-type per_tensor" > ${DIR}/output/${MODEL}/cifar100/${VERSION}/train.log 2>&1 &




	