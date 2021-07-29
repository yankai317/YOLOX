cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
python tools/train.py \
        -expn yolov3-ducha-dist \
        -n yolov3_ducha \
        -d 8 \
        -b 128 \
        --fp16 \
        -o \
        --num_machine $WORLD_SIZE \
        --machine_rank $RANK \
        --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT"