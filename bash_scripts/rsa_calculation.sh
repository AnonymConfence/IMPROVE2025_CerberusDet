
CUDA_VISIBLE_DEVICES="0" \
python anonymnet/rsa.py --img 640 --batch 1 \
--data data/voc_obj365.yaml \
--cfg anonymnet/models/yolov8x.yaml \
--single-models-cfg data/single_models_config_voc_obj365.yaml \
--name rsa_voc_obj365_v8x --exist-ok
