export PATH=/usr/local/cuda-11.6/bin:$PATH
screen -mS ms3 python train.py --outdir=./train-runs --data=./datasets/train_data1.zip --cfg=stylegan3-t --cond=True --gpus=1 --batch=32 --gamma=2 --snap=10
