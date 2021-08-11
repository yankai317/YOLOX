cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
pkill -f -9 python
python tools/train.py \
        -expn ducha/yolox-m-ducha-no-freeze \
        -n yolox-m-ducha \
        -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_m.pth.tar \
        -d 8 \
        -b 256 \
        --fp16 \
        -o \
        --num_machines $WORLD_SIZE \
        --machine_rank $RANK
bash /workspace/mnt/storage/yankai/test_cephfs/YOLOX/run_yolox_m_ducha.sh