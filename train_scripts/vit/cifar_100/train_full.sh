
DIR=/home/shkim/SSF_org/SSF
VERSION=org_full
MODEL=vit_base_patch16_224_in21k
export DIR VERSION 
mkdir -p ${DIR}/output/${MODEL}/cifar100/${VERSION}


nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4  --master_port=14655 \
	train.py /home/shkim/data/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/vit_base_patch16_224_in21k/cifar100/full \
	--save-cp-point 10 20 30 \
	--amp  --pretrained" > ${DIR}/output/${MODEL}/cifar100/${VERSION}/train.log 2>&1 &