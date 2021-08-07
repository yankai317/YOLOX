cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
python tools/train.py \
        -expn yolox-m-ducha-dist-new-mix \
        -n yolox-m-ducha \
        -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_m.pth.tar \
        -d 8 \
        -b 256 \
        --fp16 \
        -o \
        --num_machines $WORLD_SIZE \
        --machine_rank $RANK