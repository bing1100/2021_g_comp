CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.007 --workers 4 --epochs 50 --batch-size 16 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 5 --dataset pascal
