cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
pkill -f -9 python
python tools/train.py \
        -expn ducha/yolox-m-e100-new-scale-13 \
        -n yolox-m-ducha \
        -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_m.pth \
        -d 8 \
        -b 256 \
        --fp16 \
        -o \
        --cache