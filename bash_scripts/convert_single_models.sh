

python3 anonymnet/convert_to_anonymnet.py \
--weights pretrained/VOC_07_12_best_state_dict.pt \
--cfg anonymnet/models/yolov8x.yaml \
--output pretrained/yolov8x_VOC.pt \
--data data/voc_obj365.yaml \
--task_name voc


python3 anonymnet/convert_to_anonymnet.py \
--weights pretrained/OBJ365_animals_best_state_dict.pt \
--cfg anonymnet/models/yolov8x.yaml \
--output pretrained/yolov8x_obj365_animals.pt \
--data data/voc_obj365.yaml \
--task_name objects365_animals
