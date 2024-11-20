import itertools
from typing import List

from anonymnet.rsa_utils.visualizing_utils import Graph


def change_config_order(model_config):
    # [0, [15, 16], [17]] -> [0, [17], [15, 16]]
    for sub_conf in model_config:
        sub_conf[1] = sorted(sub_conf[1], key=lambda branches: len(branches))
    return model_config


def get_leaf_score(selected_tasks, task_list, dissimilarity_matrix):
    leaf_score = 0
    for i in range(len(selected_tasks)):
        orig_ind_i = task_list.index(selected_tasks[i])
        max_dissimilarity = 0
        for j in range(len(selected_tasks)):
            orig_ind_j = task_list.index(selected_tasks[j])
            diss = dissimilarity_matrix[orig_ind_i, orig_ind_j]
            if diss > max_dissimilarity:
                max_dissimilarity = diss
        leaf_score += max_dissimilarity

    return leaf_score / len(selected_tasks)


def get_graph_score(task_list, layer_ids, dissimilarity_matrix, head_ind_map, model_schedule):

    graph = Graph()

    n_layers = len(layer_ids)

    parents = [graph.last_leaf]
    available_childrens = {graph.last_leaf: task_list}
    tree_score = 0

    head_ind_map_inv = {ind: task for task, ind in head_ind_map.items()}

    schedule_map = {}
    for sc in model_schedule:  # schedule e.g.: [[4, [[16, 17], [15]]], [12, [[17], [16]]]]
        schedule_to_tasks = sc[1].copy()
        for ii, task_ind_cluster in enumerate(sc[1]):
            schedule_to_tasks[ii] = [head_ind_map_inv[ind] for ind in task_ind_cluster]
        schedule_map[sc[0]] = schedule_to_tasks

    while graph.cur_level != n_layers:
        new_level_nodes = []
        new_available_childrens = {}
        selected_ind = 0
        for parent_node in parents:
            tasks = available_childrens[parent_node].copy()
            while len(tasks) != 0:
                if graph.cur_level == n_layers - 1:
                    selected_tasks = [tasks[0]]
                elif layer_ids[graph.cur_level] in schedule_map:
                    if selected_ind < len(schedule_map[layer_ids[graph.cur_level]]) and all(
                        t in tasks for t in schedule_map[layer_ids[graph.cur_level]][selected_ind]
                    ):

                        selected_tasks = schedule_map[layer_ids[graph.cur_level]][selected_ind]
                        selected_ind += 1
                    else:
                        selected_tasks = tasks[:]
                else:
                    selected_tasks = tasks[:]

                new_leaf_name = ""
                for t in selected_tasks:
                    tasks.remove(t)
                    if graph.cur_level < n_layers - 1:
                        new_leaf_name += t[0].upper()
                    else:
                        new_leaf_name += t

                if graph.cur_level < n_layers - 1:
                    new_leaf_name += "_" + str(layer_ids[graph.cur_level + 1])

                graph.addEdge(parent_node, graph.last_leaf + 1, new_leaf_name)
                new_available_childrens[graph.last_leaf + 1] = selected_tasks
                new_level_nodes.append(graph.last_leaf + 1)
                graph.last_leaf += 1

                # get score of new graph node
                if graph.cur_level not in head_ind_map_inv:
                    print(
                        f"Get score for {selected_tasks} tasks at level {graph.cur_level}"
                        f" ({dissimilarity_matrix.shape})"
                    )
                    tree_score += get_leaf_score(selected_tasks, task_list, dissimilarity_matrix[:, :, graph.cur_level])

        graph.cur_level += 1
        parents = new_level_nodes
        available_childrens = new_available_childrens

    return graph, tree_score, model_schedule


def list_all_possible_trees(head_ind_map, branch_ids: List[int]):
    def _remove_duplicates(variants):
        final_variants = []
        for v in variants:
            if v not in final_variants:
                final_variants.append(v)

        return final_variants

    def _simplify(variants, head_ind_map):
        # Simplify [[1, [[15, 16, 17]]], [9, [[15, 16], [17]]] -> [9, [[15, 16], [17]]]
        # or [9, [[15, 16], [17]]], [11, [[15, 16]]]] -> [9, [[15, 16], [17]]]
        backbone_configuration = sorted(list(head_ind_map.values()))
        for i, schedules in enumerate(variants):
            schedules = [schedule for schedule in schedules if schedule[-1][0] != backbone_configuration]
            variants[i] = schedules
        return variants

    def _get_level_possibilittes(head_indexes):
        variants = []
        for n_combinations in range(1, len(head_indexes) + 1):
            level_combs = itertools.combinations(head_indexes, n_combinations)
            for comb in level_combs:
                variant = [sorted(list(comb))]
                left_values = [v for v in head_indexes if v not in comb]
                if len(left_values):
                    subvariants = _get_level_possibilittes(left_values)
                    variants.extend([sorted(variant[:] + v) for v in subvariants])
                else:
                    variants.append(variant)

        return _remove_duplicates(variants)

    def _get_trees(start_node, head_ind_map, branch_ids, n_splitting):

        all_variants = []
        for i in range(start_node, len(branch_ids) - 1):
            split_branch_id = branch_ids[i]
            variants = _get_level_possibilittes(head_ind_map.values())
            for var in variants:  # e.g. [[15, 16], [17]
                for mini_cluster in var:  # e.g. [15, 16]
                    for _n_splitting in range(0, min(len(mini_cluster), n_splitting)):
                        schedule = [[split_branch_id, var]]
                        if _n_splitting == 0:
                            all_variants.append(schedule)
                        else:
                            sub_head_ind_map = {k: v for k, v in head_ind_map.items() if v in mini_cluster}
                            subtrees = _get_trees(i + 1, sub_head_ind_map, branch_ids, _n_splitting)
                            all_variants.extend([schedule[:] + subschedule for subschedule in subtrees])

        # simplify schedules
        all_variants = _simplify(all_variants, head_ind_map)
        return _remove_duplicates(all_variants)

    model_schedules = []  # each element is list of elements like [0, [[15, 16], [17]]]
    n_tasks = len(head_ind_map)
    variants = _get_trees(0, head_ind_map, branch_ids, n_tasks)
    model_schedules.extend(variants)
    model_schedules = list(map(change_config_order, model_schedules))

    return model_schedules
