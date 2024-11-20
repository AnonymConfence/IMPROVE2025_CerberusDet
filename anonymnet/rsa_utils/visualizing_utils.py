import random

import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self):

        # list which stores all the set of edges
        self.visual = []
        self.labels = {}

        self.last_leaf = 0
        self.cur_level = 0
        self.labels[0] = "backbone_0"

    def addEdge(self, a, b, label_b):
        self.visual.append([a, b])
        self.labels[b] = label_b

    def visualize(self, title, out_name):
        fig = plt.figure(figsize=(20, 30), dpi=120)

        ax = plt.gca()
        ax.set_title(title, fontsize=50)

        graph = nx.DiGraph()
        graph.graph["dpi"] = 120

        graph.add_edges_from(self.visual)
        pos = hierarchy_pos(graph, 0)

        colors = []
        labels = {}
        for node, lbl in self.labels.items():
            ind = lbl.split("_")[-1]
            try:
                ind = int(ind)
            except ValueError:
                ind = len(self.labels)
            colors.append(ind)
            labels[node] = lbl.replace("_", "")

        nx.draw(
            graph,
            pos=pos,
            with_labels=False,
            node_size=[len(lbl) * 1000 for lbl in labels.values()],
            # node_shape=',',
            node_color=colors,
            cmap=plt.cm.Blues,
            arrowsize=50,
        )
        # nx.draw_networkx_edges(graph, pos)
        # nx.draw_networkx_nodes(graph, pos,
        #                 node_size=[len(lbl)*80 for lbl in labels.values()],
        #                 node_color=colors, cmap=plt.cm.Blues, alpha=0.9)

        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(graph, pos, font_size=40, bbox=label_options, labels=labels)
        # nx.draw_networkx(graph)

        fig.savefig(out_name)


def hierarchy_pos(G, root=None, width=500.0, vert_gap=100, vert_loc=0, xcenter=250):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=500.0, vert_gap=100, vert_loc=0, xcenter=250, pos=None, parent=None):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def plot_similarity(similarity, x_descriptions, y_descriptions, out_name):
    count = len(x_descriptions)

    fig = plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=1.0)
    plt.colorbar()
    plt.yticks(range(count), y_descriptions, fontsize=18)
    plt.xticks(range(count), x_descriptions, fontsize=18)
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Similarity between tasks", size=20)
    fig.savefig(f"{out_name}.png", dpi=fig.dpi)
