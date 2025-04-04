

DIR=/home/shkim/SSF_org/SSF
VERSION=probe_full_0reg
MODEL=fullbits_vit_base_patch16_224
export DIR VERSION 
mkdir -p ${DIR}/prob/cifar100/${VERSION}
MASTER_PORT=$((RANDOM % 10000 + 10000))

nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} \
	train_probe.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100 --model ${MODEL} \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${DIR}/prob/cifar100/${VERSION} \
	--register-num 8 \
	--evaluate \
	--resume /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_8reg_32_4e_4_for_last_checkpoint/fullbits_vit_base_patch16_224-full_8reg_32_4e_4_for_last_checkpoint/8reg_with_4e-4LR_32BS_110Epoch.tar \
	--pretrained" > ${DIR}/prob/cifar100/${VERSION}/train.log 2>&1 &

	# /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_8reg_32_4e_4_for_last_checkpoint/fullbits_vit_base_patch16_224-full_8reg_32_4e_4_for_last_checkpoint/8reg_with_4e-4LR_32BS_110Epoch.tar
	# /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_0reg_32_4e_4_for_last_checkpoint/fullbits_vit_base_patch16_224-full_0reg_32_4e_4_for_last_checkpoint/0reg_with_4e-4LR_32BS_110Epoch.tar
		# --evaluate \

	# --resume-mode /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_8reg_32_4e_4/fullbits_vit_base_patch16_224-full_8reg_32_4e_4/checkpoint_10.pth.tar \