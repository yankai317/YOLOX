cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
pkill -f -9 python
python tools/train.py \
        -expn coco/yolox-rex10-fpn768-e400 \
        -n yolox-rexnet10 \
        -d 8 \
        -b 256 \
        --fp16 \
        -o \
        --cache \
        --num_machine $WORLD_SIZE \
        --machine_rank $RANK \
        --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT"