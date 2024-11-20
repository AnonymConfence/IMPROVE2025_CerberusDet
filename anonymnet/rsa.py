import argparse
import copy
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from anonymnet.models.common import Concat
from anonymnet.models.net import OurAnonymModel
from anonymnet.rsa_utils.graph_utils import get_graph_score, list_all_possible_trees
from anonymnet.rsa_utils.inference_utils import (
    get_graph_computational_score,
    get_one_model_inference_time,
    load_data_for_comp_score,
)
from anonymnet.rsa_utils.similarity import get_similarity_from_rdms
from anonymnet.rsa_utils.visualizing_utils import plot_similarity
from anonymnet.utils.checks import check_file, check_requirements
from anonymnet.utils.general import check_img_size, colorstr, increment_path, set_logging
from anonymnet.utils.models_manager import ModelManager
from anonymnet.utils.torch_utils import select_device
from anonymnet.utils.train_utils import create_data_loaders
from anonymnet.val import preprocess
from loguru import logger
from tqdm import tqdm

torch.backends.cudnn.enabled = False
LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def get_single_models_params(single_models_cfg):

    with open(single_models_cfg) as f:
        single_models_info = yaml.safe_load(f)

    task_names = single_models_info["task_names"]
    single_models_pt = single_models_info["single_models"]
    configs_paths = single_models_info["configs"]

    assert (
        isinstance(task_names, list) and isinstance(single_models_pt, list) and isinstance(configs_paths, list)
    ), "Invalid single models config"
    assert len(task_names) == len(single_models_pt) == len(configs_paths), "Invalid single models config"

    configs = {task_names[i]: configs_paths[i] for i in range(len(task_names))}
    models = {task_names[i]: single_models_pt[i] for i in range(len(task_names))}

    return task_names, configs, models


def visualize(matrixes_dir, dist="pearson", feature_norm="znorm", task_list=[]):

    assert len(task_list), "Invalid visualization params"

    files = os.listdir(matrixes_dir)
    files = [f for f in files if f.endswith(".npy")]

    dist_type = [dist]
    feature_norm_type = [feature_norm]  # possible normalizations (Q,D in DDS)

    for file_name in files:
        branch_ind = file_name.split(".")[0].split("_")[-1]
        branch_file_matrix = os.path.join(matrixes_dir, file_name)
        affinity_ablation = np.load(branch_file_matrix, allow_pickle=True).item()

        for dist in dist_type:  # affinity_ablation.keys():
            for feature_norm in feature_norm_type:  # affinity_ablation[dist].keys():
                affinity_matrix = affinity_ablation[dist][feature_norm]

                # visualize affinity_matrix
                out_name = os.path.join(matrixes_dir, dist + "__" + feature_norm + "__" + branch_ind)
                plot_similarity(affinity_matrix, task_list, task_list, out_name)


def compute_networks_scores(
    task_list,
    matrixes_dir,
    opt,
    device,
    head_ind_map,  # e.g. {'furniture': 15, 'clothes': 16, 'accsess': 17}
    branch_ids,
    one_model_time,
    out_file,
    vis_dir="trees",
    dist="pearson",
    feature_norm="znorm",
):
    os.makedirs(vis_dir, exist_ok=True)

    files = os.listdir(matrixes_dir)
    files = sorted([f for f in files if f.endswith(".npy")], key=lambda x: int(x.split(".npy")[0].split("_")[-1]))

    n_layers = len(files)
    n_tasks = len(task_list)
    dissimilarity_matrix = np.zeros((n_tasks, n_tasks, n_layers), float)

    ii = 0
    layer_ids = []
    for file_name in files:
        laeyr_ind = file_name.split(".")[0].split("_")[-1]
        layer_ids.append(laeyr_ind)
        branch_file_matrix = os.path.join(matrixes_dir, file_name)
        affinity_ablation = np.load(branch_file_matrix, allow_pickle=True).item()

        affinity_matrix = affinity_ablation[dist][feature_norm]
        if affinity_matrix.shape[0] > n_tasks:
            affinity_matrix = affinity_matrix[:n_tasks, :n_tasks]
        dissimilarity_matrix[:, :, ii] = affinity_matrix
        ii += 1

    dissimilarity_matrix = 1 - dissimilarity_matrix

    layer_ids = [0] + list(map(int, layer_ids))
    all_schedules = list_all_possible_trees(head_ind_map, branch_ids)
    print(f"Found {len(all_schedules)} variants")

    dict_results = {"config": [], "rsa_score": [], "comp_score": []}

    if os.path.exists(out_file):
        df = pd.read_csv(out_file, names=["config", "rsa_score", "comp_score"], header=None)
        dict_results["config"] = [eval(el) for el in df["config"].tolist()[1:]]
        dict_results["rsa_score"] = [eval(el) for el in df["rsa_score"].tolist()[1:]]
        dict_results["comp_score"] = [eval(el) for el in df["comp_score"].tolist()[1:]]

    # load data once
    model_manager, val_loaders = load_data_for_comp_score(opt.hyp, opt, device)

    all_schedules = [conf for conf in all_schedules if conf not in dict_results["config"]]
    n_variants = len(all_schedules)

    for ii in range(len(all_schedules)):

        selected_conf = all_schedules[ii]

        print(f"Processing {ii}/{n_variants}: {selected_conf}")
        # get model architecture scores
        graph, tree_score, model_schedule = get_graph_score(
            task_list, layer_ids, dissimilarity_matrix, head_ind_map, selected_conf
        )

        cur_conf = [copy.deepcopy(sub_conf) for sub_conf in selected_conf]
        # NOTE: cur_conf will be changed here
        perf_score = get_graph_computational_score(cur_conf, model_manager, val_loaders, opt, device, one_model_time)

        print(f"{ii}/{n_variants}", model_schedule, "rsa score, comp score (lower => better): ", tree_score, perf_score)

        dict_results["config"].append(selected_conf)
        dict_results["rsa_score"].append(tree_score)
        dict_results["comp_score"].append(perf_score)

        tree_score, perf_score = f"{tree_score:.3f}", f"{perf_score:.3f}"
        graph.visualize(
            f"RSA, Comp scores: {tree_score}, {perf_score} \n({model_schedule})",
            f"{vis_dir}/tree_{tree_score}_{perf_score}.png",
        )

        # save scores
        res = pd.DataFrame.from_dict(dict_results).sort_values(by=["rsa_score"])
        res.to_csv(out_file, index=False)


def compute_task_groupping(scores_file, n_tasks, one_model_time, max_time=None):
    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"File {scores_file} not found")

    dict_results = {}
    df = pd.read_csv(scores_file, names=["config", "rsa_score", "comp_score"], header=None)

    dict_results["config"] = [eval(el) for el in df["config"].tolist()[1:]]
    dict_results["rsa_score"] = [eval(el) for el in df["rsa_score"].tolist()[1:]]
    dict_results["comp_score"] = [eval(el) for el in df["comp_score"].tolist()[1:]]

    results = list(zip(dict_results["rsa_score"], dict_results["comp_score"], dict_results["config"]))
    if max_time is not None:
        filter_time_score = max_time / (one_model_time * n_tasks)
    else:
        filter_time_score = 1

    results = list(filter(lambda res: res[1] < filter_time_score, results))
    print(
        f"Max available computational score: {filter_time_score} "
        f"Left variants: {len(results)}/{len(dict_results['config'])}"
    )

    if len(results) == 0:
        return None

    results = sorted(results, key=lambda res: (res[0] + res[1]) / 2)
    best = results[0]

    print(
        f"From {len(dict_results['config'])} configurations found best {best[2]} "
        f"with rsa score {best[0]} and comp score {best[1]} (mean {(best[0] + best[1])/2})"
    )

    return results[0][2]


def compute_DDS(out_dir, feature_file):
    """
    Computing Duality Diagram Similarity between tasks
    """
    os.makedirs(out_dir, exist_ok=True)

    feature_per_task: Dict[str, dict] = np.load(feature_file, allow_pickle=True).item()
    task_list = list(feature_per_task.keys())
    layers_keys = list(feature_per_task[task_list[0]].keys())
    n_images = len(feature_per_task[task_list[0]][layers_keys[0]])

    logger.info(f"Loaded features for {task_list} tasks, {len(layers_keys)} branches, {n_images} images")

    dist_type = ["pearson", "euclidean", "cosine"]
    # kernel_type = ['rbf','lap','linear']
    feature_norm_type = ["None", "centering", "znorm"]  # possible normalizations (Q,D in DDS)

    for branch_ind in layers_keys:
        save_path = os.path.join(out_dir, f"dds_{branch_ind}.npy")
        if os.path.exists(save_path):
            logger.warning(f"{save_path} exists. Skip calculation..")
            continue

        affinity_ablation = {}
        for dist in dist_type:
            affinity_ablation[dist] = {}
            for feature_norm in feature_norm_type:
                affinity_matrix = np.zeros((len(task_list), len(task_list)), float)

                method = dist + "__" + feature_norm
                start = time.time()
                for index1, task1 in enumerate(task_list):
                    for index2, task2 in enumerate(task_list):
                        if index1 > index2:
                            continue
                        task_1_branch_features = np.squeeze(np.array(feature_per_task[task1][branch_ind]), 1)
                        task_2_branch_features = np.squeeze(np.array(feature_per_task[task2][branch_ind]), 1)
                        affinity_matrix[index1, index2] = get_similarity_from_rdms(
                            task_1_branch_features, task_2_branch_features, dist, feature_norm, flattened=False
                        )
                        affinity_matrix[index2, index1] = affinity_matrix[index1, index2]
                end = time.time()
                print("Method is ", method)
                print(f"Time taken is {end - start} sec")
                affinity_ablation[dist][feature_norm] = affinity_matrix

        np.save(save_path, affinity_ablation)
        logger.info(f"Saved {save_path}")


def collect_features(opt, device, out_file, half=True, num_images_from_tasks=100) -> Tuple[Dict[str, int], List[int]]:
    """
    Collect features from single trained models from all layers which can be shared(!).
    For calculation from each single dataset will be taken `num_images_from_tasks` images.
    """

    TASK_NAMES, CONFIGS, SINGLE_MODELS = get_single_models_params(opt.single_models_cfg)

    cfg = opt.cfg
    model_manager = ModelManager(opt.hyp, opt, RANK, LOCAL_RANK)
    hyp = model_manager.hyp

    shared_model, _ = model_manager.load_model(cfg, device)
    assert isinstance(shared_model, OurAnonymModel)

    gs = max(int(shared_model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs)  # verify imgsz is gs-multiple

    tasks_models = {}
    for i, task_name in enumerate(model_manager.task_ids):
        if opt.cfg == CONFIGS[task_name]:
            peeled_model, _ = shared_model.peel(model_manager.task_ids[i : (i + 1)], device)
        else:
            # build model from different config
            # TMP solution as accssories model was trained with different backbone config
            logger.warning(f"Use different config for {task_name} task")
            peeled_model, _ = model_manager.load_model(CONFIGS[task_name], device)[0].peel(
                model_manager.task_ids[i : (i + 1)], device
            )

        print(peeled_model.info())
        print("\n")
        task_model = torch.load(SINGLE_MODELS[task_name], map_location=device)
        _, loaded = model_manager.from_ckpt(task_model, peeled_model)
        assert loaded

        # using fuse model to avoid batchnorm affect
        peeled_model = peeled_model.float().fuse().eval()

        if half:
            peeled_model.half()
        tasks_models[task_name] = peeled_model

    # Dataloaders
    # load all images with same shapes
    _, val_loaders, _, _ = create_data_loaders(
        model_manager.data_dict,
        RANK,
        WORLD_SIZE,
        opt,
        hyp,
        gs,
        imgsz,
        val_pad=0.0,
        val_rect=False,
        skip_train_load=True,
    )

    assert len(val_loaders) == model_manager.num_tasks
    assert len(tasks_models) == len(val_loaders)

    feature_per_task: Dict[str, dict] = {task: {} for task in model_manager.task_ids}

    for task_i, task in enumerate(model_manager.task_ids):

        val_loader = val_loaders[task_i]

        seen = 0
        s = ("%20s" * 1 + "%11s" * 1) % ("Task", "Images")
        for batch_i, batch in enumerate(tqdm(val_loader, desc=s)):
            batch = preprocess(batch, half, device)

            seen += batch["img"].shape[0]
            for task_name, peeled_model in tasks_models.items():
                _ = peeled_model(batch["img"], retain_all=True)

                feature_indexes = peeled_model.rep_tensors.keys()
                feature_indexes = [ind for ind in feature_indexes if ind not in [peeled_model.heads[task_name]]]

                for ind in feature_indexes:
                    if ind not in feature_per_task[task_name]:
                        feature_per_task[task_name][ind] = []

                    block_out = peeled_model.rep_tensors[ind]
                    if isinstance(block_out, list):
                        assert ind == 0
                        block_out = block_out[-1]  # last backbone layer output

                    feature_per_task[task_name][ind].append(block_out.detach().cpu().numpy())

            # compute similarity for first num_images from each task specific dataset
            if seen >= num_images_from_tasks:
                break

        pf = "%20s" * 1 + "%11i" * 1  # print format
        print(pf % (task, seen))

    # save features into file
    np.save(out_file, feature_per_task, allow_pickle=True)
    logger.info(f"Features are saved into {out_file}")

    possible_branch_ids: List[int] = []
    for ind, block in enumerate(peeled_model.controllers):
        if ind == peeled_model.heads[task_name]:
            continue
        childrens = block.children_indices
        wo_weights = [isinstance(peeled_model.blocks[ch], (Concat, nn.Upsample)) for ch in childrens]
        if all(wo_weights):
            continue
        possible_branch_ids.append(ind)

    return shared_model.heads, possible_branch_ids


def get_model_heads(opt) -> Tuple[Dict[str, int], List[int]]:
    cfg = opt.cfg
    model_manager = ModelManager(opt.hyp, opt, RANK, LOCAL_RANK)

    shared_model, _ = model_manager.load_model(cfg, device, verbose=False)
    assert isinstance(shared_model, OurAnonymModel)

    task_name = model_manager.task_ids[0]
    peeled_model, _ = shared_model.peel(task_name, device)

    possible_branch_ids = []

    for ind, block in enumerate(peeled_model.controllers):
        if ind == peeled_model.heads[task_name]:
            continue
        childrens = block.children_indices
        wo_weights = [isinstance(peeled_model.blocks[ch], (Concat, nn.Upsample)) for ch in childrens]
        if all(wo_weights):
            continue
        possible_branch_ids.append(ind)

    return shared_model.heads, possible_branch_ids


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5x6.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default="data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument(
        "--single-models-cfg", type=str, default="data/hyps/single_models_config.yaml", help="single models config path"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--workers", type=int, default=8, help="maximum number of dataloader workers")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument("--experiment_name", type=str, default="exp_rsa", help="MlFlow experiment name")
    parser.add_argument(
        "--mlflow-url",
        type=str,
        default=None,
        help="Param for mlflow.set_tracking_uri(), may be 'local'",
    )
    parser.add_argument(
        "--use-multi-labels", action="store_true", help="Loading multiple labels for boxes, if available"
    )
    parser.add_argument("--use-soft-labels", action="store_true", help="Class probability based on annotation votes")
    parser.add_argument("--labels-from-xml", action="store_true", help="Load labels from xml files")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr("DDS: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
        check_requirements(exclude=["thop"])

    opt.weights = ""
    opt.evolve = False
    opt.hyp = None
    opt.epochs = 0
    opt.noval = False
    opt.cache_images = False
    opt.data, opt.cfg = check_file(opt.data), check_file(opt.cfg)  # check files
    opt.single_models_cfg = check_file(opt.single_models_cfg)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    print("Save dir: ", opt.save_dir)
    with open(opt.single_models_cfg) as f:
        task_list = yaml.safe_load(f)["task_names"]

    assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
    device = select_device(opt.device, batch_size=opt.batch_size)

    cfg_name = opt.cfg.split(os.sep)[-1].split(".")[0]
    features_file = os.path.join(opt.save_dir, f"{cfg_name}_branch_features_not_rect_100.npy")
    if not os.path.exists(features_file):
        head_ind_map, branch_ids = collect_features(opt, device, features_file, num_images_from_tasks=100)
    else:
        head_ind_map, branch_ids = get_model_heads(opt)

    print(f"Possible branch ids: {branch_ids}")

    rdds_dir = os.path.join(opt.save_dir, "matrixes")
    if not os.path.exists(rdds_dir):
        compute_DDS(rdds_dir, features_file)
        visualize(rdds_dir, task_list=task_list)

    TASK_NAMES, _, SINGLE_MODELS = get_single_models_params(opt.single_models_cfg)
    # in ms for batch 1 and half model
    one_model_time = get_one_model_inference_time(opt.hyp, opt, device, SINGLE_MODELS, task=TASK_NAMES[0], half=True)
    print(f"One model with batch {opt.batch_size} takes {one_model_time} ms")

    scores_file = os.path.join(opt.save_dir, "scores.csv")
    vis_dir = os.path.join(opt.save_dir, "trees")
    compute_networks_scores(
        task_list, rdds_dir, opt, device, head_ind_map, branch_ids, one_model_time, scores_file, vis_dir=vis_dir
    )

    for time_scale in [1.85, 2, 2.25, 2.5, 2.75, 3]:
        mdoel_config = compute_task_groupping(
            scores_file, n_tasks=len(task_list), one_model_time=one_model_time, max_time=one_model_time * time_scale
        )
