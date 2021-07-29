cd /workspace/mnt/storage/yankai/test_cephfs/YOLOX
python setup.py develop  # or  python3 setup.py develop
python tools/train.py -n yolov3_ducha -d 8 -b 64 --fp16 -o -c /workspace/mnt/storage/yankai/test_cephfs/YOLOX/pretrain/yolox_darknet53.47.3.pth.tar