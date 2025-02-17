#python -m torch.distributed.run --standalone --nproc_per_node=gpu   ../tools/train.py config_ci2pvit_imagenet.py
CUDA_VISIBLE_DEVICES=4,5 bash ../tools/dist_train.sh config_myvitvar_imagenet.py 1 --resume /home/zxf/code/checkpoint/myvar_epoch_54.pth  --work-dir work_imagenet


