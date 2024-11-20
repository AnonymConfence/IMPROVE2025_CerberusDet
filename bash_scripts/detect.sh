
python3 anonymnet/detect.py \
--img 640 --half \
--weights 'weights/voc_obj365_v8x_best.pt' \
--source data/images \
--device 0 \
--hide-conf
