

DIR=/home/shkim/SSF_org/SSF
VERSION=pertokenA_pertokenAG_5e_5_resumeWith_0reg_32_4e_4_10ep
MODEL=fourbits_vit_base_patch16_224
export DIR VERSION MODEL
mkdir -p ${DIR}/output/${MODEL}/cifar100/${VERSION}
MASTER_PORT=$((RANDOM % 10000 + 10000))

nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} \
	train.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100 --model ${MODEL} \
    --batch-size 32 --epochs 100 \
	--opt adamw --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${DIR}/output/${MODEL}/cifar100/${VERSION} \
	--register-num 0 --freeze-reg \
	--a-quant-type per_token \
	--ag-quant-type per_token \
	--resume-mode /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_0reg_32_4e_4_for_10checkpoint/fullbits_vit_base_patch16_224-full_0reg_32_4e_4_for_10checkpoint/checkpoint_10.pth.tar \
	--log-wandb --experiment ${MODEL}-${VERSION} \
	--pretrained" > ${DIR}/output/${MODEL}/cifar100/${VERSION}/train.log 2>&1 &

	#	--resume-mode /home/shkim/QT/deit/output/finetune/vit_base_finetuning_in1k_fullprecision_8register_for_checkpoint/fullprecision_8reg_checkpoint-625_acc-69.96.pth.tar \
#/home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_8reg_32_4e_4/fullbits_vit_base_patch16_224-full_8reg_32_4e_4/checkpoint_10.pth.tar
# /home/shkim/SSF_org/SSF/output/fullbits_vit_base_patch16_224/cifar100/full_0reg_128_4e_4_for_10checkpoint/fullbits_vit_base_patch16_224-full_0reg_128_4e_4_for_10checkpoint/checkpoint_10.pth.tar

		

		
