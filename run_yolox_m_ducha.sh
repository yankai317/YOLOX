cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
python tools/train.py \
        -expn ducha/yolox-m-e300-lr3e-3-rs_8_13-dg5-sc5_15-we_10-na_10-mlr_01-bs-512 \
        -n yolox-m-ducha \
        -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_m.pth.tar \
        -d 8 \
        -b 512 \
        --fp16 \
        -o \
        --num_machines $WORLD_SIZE \
        --machine_rank $RANK
pkill -f -9 python
bash /workspace/mnt/storage/yankai/test_cephfs/YOLOX/run_yolox_m_ducha.sh