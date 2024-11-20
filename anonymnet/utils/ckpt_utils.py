from anonymnet.models.net import OurAnonymModel
from loguru import logger


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values

    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def dict_to_anonymnet(loaded_dict: dict, model: OurAnonymModel):
    """Map yolo state_dict to anonymnet state_dict

    :param loaded_dict: state_dict of original yolo model
    :param model: an instance of anonymnet model
    :return: state_dict to initialize anonymnet with weights from loaded_dict
    """

    anonymnet_state_dict = model.state_dict()
    old_head_n = None

    heads_nums = list(model.heads.values())

    for k, v in loaded_dict.items():
        if ".dfl" in k:
            old_head_n = k.split(".")[1]

    blocks = model.blocks
    key_prefix = "blocks"

    yolo_to_anonymnet_inds = {}
    # map old indexes to new blocks
    for ind, block in enumerate(blocks):
        if ind == 0:
            next_block_example = blocks[1]
            # backbone
            for old_i in range(next_block_example.i):
                yolo_to_anonymnet_inds[old_i] = 0
            continue

        cur_block = block
        yolo_to_anonymnet_inds[cur_block.i] = ind

    # map backbone, neck and head keys
    new_dict = {}
    for k, v in loaded_dict.items():
        if old_head_n is not None and f"model.{old_head_n}." in k:
            # heads
            for i in heads_nums:
                new_anonymnet_key = f"{key_prefix}.{i}." + ".".join(k.split(".")[2:])
                new_dict[new_anonymnet_key] = v
        else:
            yolov8_i = int(k.split(".")[1])  # model.24.m.0.bias -> 24

            new_anonymnet_key = None
            if yolov8_i in yolo_to_anonymnet_inds and yolo_to_anonymnet_inds[yolov8_i] == 0:
                # backbone
                new_anonymnet_key = f"{key_prefix}.0.{k}"
            elif yolov8_i in yolo_to_anonymnet_inds:
                # neck
                anonymnet_block_i = yolo_to_anonymnet_inds[yolov8_i]
                if isinstance(model, OurAnonymModel):
                    new_anonymnet_key = f"{key_prefix}.{anonymnet_block_i}." + ".".join(k.split(".")[2:])
                else:
                    new_anonymnet_key = [
                        f"{key_prefix}.{anonymnet_block_i}.path.{task_ind}." + ".".join(k.split(".")[2:])
                        for task_ind in range(len(model.tasks))
                    ]  # type: ignore

            if new_anonymnet_key is None:
                logger.warning(f"Yolo key has not been mapped: {k}")
                continue

            new_anonymnet_key = (
                new_anonymnet_key if isinstance(new_anonymnet_key, list) else [new_anonymnet_key]  # type: ignore
            )

            for anonymnet_key in new_anonymnet_key:
                if anonymnet_key not in anonymnet_state_dict:
                    logger.warning(f"\tKey {anonymnet_key} has not been found in the current net model dict")
                    continue

                if anonymnet_state_dict[anonymnet_key].shape != v.shape:
                    logger.warning(
                        f"\tMismatched shapes for {anonymnet_key}: "
                        f"old {v.shape} and new {anonymnet_state_dict[anonymnet_key].shape}"
                    )
                    continue

                new_dict[anonymnet_key] = v

    return new_dict
