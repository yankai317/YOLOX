cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
python tools/train.py \
        -expn coco/yolox-m \
        -n yolox-m \
        -d 8 \
        -b 128 \
        --fp16 \
        -o \
        --cache