from typing import Tuple

import torch
from anonymnet.models.experimental import attempt_load
from anonymnet.train import LOCAL_RANK, RANK, WORLD_SIZE
from anonymnet.utils.general import check_img_size
from anonymnet.utils.models_manager import ModelManager
from anonymnet.utils.torch_utils import time_sync
from anonymnet.utils.train_utils import create_data_loaders
from anonymnet.val import preprocess
from tqdm import tqdm


def cal_inference_time(val_loader, model, device, title, half, warmup=5, max_steps=500) -> Tuple[int, float]:
    seen, inf_t = 0, 0
    s = ("%20s" * 1 + "%11s" * 2) % ("Task", "Images", "Mean time")
    for batch_i, batch in enumerate(tqdm(val_loader, desc=s)):
        batch = preprocess(batch, half, device)

        t_ = time_sync()
        _ = model(batch["img"])
        t = time_sync()

        if batch_i >= warmup:
            seen += batch["img"].shape[0]
            inf_t += t - t_

        if batch_i >= max_steps:
            # to speed up calculations
            break

    pf = "%20s" * 1 + "%11i" * 1 + "%11.3g" * 1  # print format
    print(pf % (title, seen, inf_t / seen * 1e3))

    return seen, inf_t


def get_one_model_inference_time(hyps, opt, device, SINGLE_MODELS, task, half=True) -> float:

    model_manager = ModelManager(hyps, opt, RANK, LOCAL_RANK)
    hyp = model_manager.hyp

    task_model = attempt_load(
        SINGLE_MODELS[task], map_location=device, mlflow_url=opt.mlflow_url
    )  # load FP32 fused model

    if half:
        task_model.half()

    gs = max(int(task_model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs)  # verify imgsz is gs-multiple

    _, val_loaders, _, _ = create_data_loaders(
        model_manager.data_dict, RANK, WORLD_SIZE, opt, hyp, gs, imgsz, skip_train_load=True
    )
    task_ind = model_manager.task_ids.index(task)

    seen, inf_t = cal_inference_time(val_loaders[task_ind], task_model, device, task, half)

    return inf_t / seen * 1e3


def load_data_for_comp_score(hyps, opt, device):

    cfg = opt.cfg
    model_manager = ModelManager(hyps, opt, RANK, LOCAL_RANK)
    hyp = model_manager.hyp
    shared_model, _ = model_manager.load_model(cfg, device, verbose=False)

    gs = max(int(shared_model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs)  # verify imgsz is gs-multiple

    _, val_loaders, _, _ = create_data_loaders(
        model_manager.data_dict, RANK, WORLD_SIZE, opt, hyp, gs, imgsz, skip_train_load=True
    )

    del shared_model

    return model_manager, val_loaders


def get_graph_computational_score(model_schedule, model_manager, val_loaders, opt, device, one_model_time, half=True):

    cfg = opt.cfg
    shared_model, _ = model_manager.load_model(cfg, device, verbose=False)

    model_head_ids = sorted(list(shared_model.heads.values()))
    shared_model.sequential_split(model_schedule, device)

    print(f"model_schedule {model_schedule}: ")
    print(shared_model.info())

    shared_model = shared_model.float().fuse().eval()

    if half:
        shared_model.half()

    inf_t = 0
    total_seen = 0
    for task_i, (task, val_loader) in enumerate(zip(model_manager.task_ids, val_loaders)):

        warmup_steps = 0 if task_i != 0 else 5
        seen, t = cal_inference_time(val_loader, shared_model, device, task, half, warmup_steps)
        total_seen += seen
        inf_t += t

    mean_model_time = inf_t / total_seen * 1e3
    print(f"Model with batch {opt.batch_size} takes {mean_model_time} ms")

    del shared_model
    torch.cuda.empty_cache()
    return mean_model_time / (one_model_time * len(model_head_ids))
