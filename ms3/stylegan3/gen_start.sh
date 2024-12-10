export PATH=/usr/local/cuda-11.6/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
screen -mS ms3 python gen_images.py --outdir=out/61 --trunc=1 --seeds=0-50000 --label=6,1 --network=train-runs/00000-stylegan3-t-train_data1-gpus1-batch32-gamma2/network-snapshot-001600.pkl
