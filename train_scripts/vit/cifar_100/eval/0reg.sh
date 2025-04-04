

DIR=/home/shkim/SSF_org/SSF
VERSION=test
MODEL=eightbits_vit_base_patch16_224
export DIR VERSION MODEL
mkdir -p ${DIR}/output/${MODEL}/cifar100/eval/${VERSION}
MASTER_PORT=$((RANDOM % 10000 + 10000))

nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} \
	train.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100 --model ${MODEL} \
    --batch-size 32 --epochs 100 \
	--opt adamw --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${DIR}/output/${MODEL}/cifar100/eval/${VERSION} \
	--register-num 8 \
	--a-quant-type per_token \
	--evaluate \
	--resume /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_8reg_32_4e_4_for_last_checkpoint/fullbits_vit_base_patch16_224-full_8reg_32_4e_4_for_last_checkpoint/checkpoint-92.pth.tar \
	--pretrained" > ${DIR}/output/${MODEL}/cifar100/eval/${VERSION}/train.log 2>&1 &



		

		
