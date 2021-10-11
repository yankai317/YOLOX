cd /workspace/mnt/storage/yankai/source/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
python tools/train.py \
        -expn ducha/yolox-m \
        -n yolox-m-ducha \
        -c /workspace/mnt/storage/yankai/source/YOLOX/YOLOX_outputs/coco/yolox-m/best_ckpt.pth \
        -d 8 \
        -b 256 \
        --fp16 \
        -o \
        --num_machine $WORLD_SIZE \
        --machine_rank $RANK \
        --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT"