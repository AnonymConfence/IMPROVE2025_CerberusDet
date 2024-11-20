
CUDA_VISIBLE_DEVICES="0" \
python3 anonymnet/train.py \
--img 640 --batch 32 \
--data data/voc_obj365.yaml \
--weights pretrained/yolov8x_state_dict.pt \
--cfg anonymnet/models/yolov8x_voc_obj365.yaml \
--hyp data/hyps/hyp.evolved.voc_obj365.yaml \
--name voc_obj365_v8x_evovled \
--mlflow-url localhost \
--patience 10
