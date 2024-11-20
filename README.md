
### OurAnonymModel: Multi-task Learning for Multi-dataset Object Detection
[[`Paper`](https://arxiv.org/abs/)]

---

The code is based on:

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv5](https://github.com/ultralytics/yolov5)

### Install

[**Python>=3.8.0**](https://www.python.org/) is required.
```bash
$ git clone
$ pip install -e .
```

### Docker

Clone repo into anonymnet_repo/ dir.

Change local paths to data in [docker-compose.yml](docker-compose.yml) file (`volumes`) and run the docker:

```bash
sudo docker-compose up -d
sudo docker attach anonymnet_repo_anonymnet_1
```

### Data

- Use script [voc.py](data/scripts/voc.py) to download VOC dataset

For information about the VOC dataset and its creators, visit the [PASCAL VOC dataset website](http://host.robots.ox.ac.uk/pascal/VOC/).
- Use script [objects365_animals.py](data/scripts/objects365_animals.py) to download part of Objects365 dataset with 19 categories, used in the paper.
```
['Monkey', 'Rabbit', 'Yak', 'Antelope', 'Pig',  'Bear', 'Deer', 'Giraffe', 'Zebra', 'Elephant',
'Lion', 'Donkey', 'Camel', 'Jellyfish', 'Other Fish', 'Dolphin', 'Crab', 'Seal', 'Goldfish']
```
The Objects365 dataset is available for the academic purpose only. For information about the dataset and its creators, visit the [Objects365 dataset website](https://www.objects365.org/).

**IMPORTANT:** If some patch tar.gz archives are still present in the `Objects365_part/tmp_images` directory, it means they were not fully downloaded. Please restart the script for missed patches to obtain all subset images.

**ONLY FOR REVIEWERS:** You can download archives with data from [link](https://drive.google.com/drive/folders/1cykS3vQ0VUqIOrMugLXvegkdV-5vi7XW?usp=sharing):

### Train

- Download pretrained on COCO [yolov8x_state_dict.pt weights](https://drive.google.com/drive/folders/1I6fbX7fV4zjxjqOcKj1aHOIIybd26gj6?usp=sharing) into `pretrained` folder.
- Run train process with 1 GPU
```bash
$ python3 anonymnet/train.py \
--img 640 --batch 32 \
--data data/voc_obj365.yaml \
--weights pretrained/yolov8x_state_dict.pt \
--cfg anonymnet/models/yolov8x_voc_obj365.yaml \
--hyp data/hyps/hyp.voc_obj365.yaml \
--name voc_obj365_v8x --device 0
```
- OR run train process with several GPUs (batch size will be divided):
```bash
$ CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node 4 anonymnet/train.py \
--img 640 --batch 128 \
--data data/voc_obj365.yaml \
--weights pretrained/yolov8x_state_dict.pt \
--cfg anonymnet/models/yolov8x_voc_obj365.yaml \
--hyp data/hyps/hyp.voc_obj365.yaml \
--name voc_obj365_v8x \
--sync-bn
```
By default logging will be done with tensorboard, but you can use mlflow if set --mlflow-url, e.g. `--mlflow-url localhost`.

<details>
<summary>OurAnonymModel model config details </summary>

Example of the model's config for 2 tasks: [yolov8x_voc_obj365.yaml](anonymnet/models/yolov8x_voc_obj365.yaml)

- The model config is based on yolo configs, except that the `head` is divided into two sections (`neck` and `head`)
- The layers of the `neck` section can be shared between tasks or be unique
- The `head` section defines what the head will be for all tasks, but each task will always have its own unique parameters
- The `from` parameter of the first neck layer must be a positive ordinal number, specifying from which layer, starting from the beginning of the entire architecture, to take features.
- The `anonymnet` section is optional and defines the architecture configuration for determining the neck layers to be shared among tasks. If not specified, all layers will be shared among tasks, and only the heads will be unique.
- The OurAnonymModel configuration is constructed as follows:<br>
  `anonymnet: List[OneBranchConfig]`, where<br>
  &nbsp; `OneBranchConfig = List[anonymnet_layer_number, SharedTasksConfig]`, where<br>
  &nbsp; &nbsp; &nbsp; `anonymnet_layer_number` - the layer number (counting from the end of the backbone) after which branching should occur<br>
  &nbsp; &nbsp; &nbsp; `SharedTasksConfig = List[OneBranchGroupedTasks]`, where<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `OneBranchGroupedTasks = [number_of_task1_head, number_of_task2_head, ...]` - the task head numbers (essentially task IDs) that should be in the same branch and share layers thereafter<br><br>
  The head numbers will correspond to tasks according to the sequence in which they are listed in the data configuration.<br><br>
  Example for YOLO v8x:<br>
  `[[2, [[15], [13, 14]]], [6, [[13], [14]]]]` - configuration for 3 tasks. Task id=15 will have all task-specific layers, starting from the 3rd. Tasks id=13, id=14 will share layers 3-6, then after the 6th, they will have their own separate branches with all layers.

</details>

### Evaluation

- Download OurAnonymModel checkpoint (see below)
- Run script [bash_scripts/val.sh](bash_scripts/val.sh)

### Inference

- Download OurAnonymModel checkpoint trianed on VOC and part of Objects 365 datasets (see below) into `weights` folder
- Run script [bash_scripts/detect.sh](bash_scripts/detect.sh)

### Pretrained Checkpoints

| Model                                                                                                                                 |  Train set               | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>V100 b32, fp16<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|---------------------------------------------------------------------------------------------------------------------------------------|-------------------------|-----------------------|----------------------|-------------------|--------------------------------------|--------------------|------------------------|
| YOLOv8x [VOC_07_12_best_state_dict.pt](https://drive.google.com/drive/folders/1I6fbX7fV4zjxjqOcKj1aHOIIybd26gj6?usp=sharing)          | VOC                     | 640                   | 0.758                | 0.916             | 5.6                                  | 68                 | 257.5                  |
| YOLOv8x [OBJ365_animals_best_state_dict.pt](https://drive.google.com/drive/folders/1I6fbX7fV4zjxjqOcKj1aHOIIybd26gj6?usp=sharing)     | Objects365_animals      | 640                   | 0.43                 | 0.548             | 5.6                                  | 68                 | 257.5                  |
| OurAnonymModel_v8x [voc_obj365_v8x_best.pt](https://drive.google.com/drive/folders/1lBJ-mJyJTDBg9M9U99DxEbu6gOOaDFuo?usp=drive_link ) | VOC, Objects365_animals | 640                   | 0.751, 0.432         | 0.918, 0.556      | 7.2                                  | 105                | 381.3                  |

YOLOv8x models were trained with the the commit: https://github.com/ultralytics/ultralytics/tree/2bc36d97ce7f0bdc0018a783ba56d3de7f0c0518

<details>

Command:
```bash
yolo task=detect mode=train \
model=yolov8x.pt \
data=data/voc.yaml \
name=VOC_07_12 \
epochs=100 \
batch=32 \
imgsz=640 \
optimizer=SGD \
lr0=0.00309 \
lrf=0.0956 \
momentum=0.952 \
weight_decay=0.00037 \
warmup_epochs=2.04 \
warmup_momentum=0.898 \
warmup_bias_lr=0.0502 \
hsv_h=0.0124 \
hsv_s=0.696 \
hsv_v=0.287 \
degrees=0.299 \
translate=0.211 \
scale=0.846 \
shear=0.717 \
perspective=0.0 \
flipud=0.00983 \
fliplr=0.5 \
mosaic=1.0 \
mixup=0.285 \
copy_paste=0.0 \
device=0 \
v5loader=True \
cache=ram \
patience=15 \
half=True
```
</details>

### Hyperparameter Evolution

See the launch example in the  [bash_scripts/evolve.sh](bash_scripts/evolve.sh).

<details>
<summary>Notes</summary>

- To evolve hyperparameters specific to each task, specify initial parameters separately per task and append `--evolve_per_task`
- To evolve specific set of hyperparameters, specify their names separated by comma via the `--params_to_evolve` argument, e.g. `--params_to_evolve 'box,cls,dfl'`
- Use absolute paths to configs.
- Specify search algorith via `--evolver`. You can use the search algorithms of the [ray](https://docs.ray.io/en/latest/index.html) library (see available values here: [predefined_evolvers.py](anonymnet/evolvers/predefined_evolvers.py)), or `'yolov5'`

</details>

### RSA

- Download YOLOv8 models trained for VOC(`VOC_07_12_best_state_dict.pt`) and Objects365_animals(`OBJ365_animals_best_state_dict.pt`) datasets from [link](https://drive.google.com/drive/folders/1I6fbX7fV4zjxjqOcKj1aHOIIybd26gj6?usp=sharing) into `pretrained` folder.
- Convert models to the anonymnet checkpoints using script [bash_scripts/convert_single_models.sh](bash_scripts/convert_single_models.sh)
- Run rsa script [bash_scripts/rsa_calculation.sh](bash_scripts/rsa_calculation.sh)

### License
OurAnonymModel is released under the MIT license.

See the file [LICENSE](LICENSE.txt) for more details.

### Citing

If you use our models, code or dataset, we kindly request you to cite the following paper and give repository a ⭐

```bibtex
@article{anonymnet,
   Author = {},
   Title = {},
   Year = {2024},
   Eprint = {arXiv:},
}
```
