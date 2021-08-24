cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
sleep 30
pkill -f -9 python
python tools/train.py \
        -expn ducha/yolox-s-ducha-100_no-mix_rd-10-13_scale_05-15 \
        -n yolox-s-ducha \
        -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_s.pth.tar \
        -d 8 \
        -b 256 \
        --fp16 \
        -o
bash /workspace/mnt/storage/yankai/test_cephfs/YOLOX/run_yoloxm.sh