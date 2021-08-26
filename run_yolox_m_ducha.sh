cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
python tools/train.py \
        -expn ducha/yolox-m-e300-lr3e-3-rs_8_13-dg5-sc5_15-we_10-na_10-mlr_01-bs-512 \
        -n yolox-m-ducha \
        -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_m.pth.tar \
        -d 8 \
        -b 256 \
        --fp16 \
        -o \
        --num_machine $WORLD_SIZE \
        --machine_rank $RANK \
        --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT"