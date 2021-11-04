import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
import networkx as nx
import numpy as np
import copy as cp



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, paths, index, sequences, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['paths']: paths})
    feed_dict.update({placeholders['index']: index})
    feed_dict.update({placeholders['sequences']: sequences})
    return feed_dict



def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def edge_convert_node(graph):
    num_edges = nx.number_of_edges(graph)
    adj = np.zeros([num_edges, num_edges], dtype=np.int)

    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])


# def k_shortest_paths(graph, source, target, k):
#     all_path = nx.all_simple_paths(graph, source, target)
#     paths = []
#     count = 0
#     for i in all_path:
#         count += 1
#         paths.append(i)
#         if count % k == 0:
#             break
#     return paths

def k_shortest_paths(G, source, target, k=1, weight='weight'):
    # G is a networkx graph.
    # source and target are the labels for the source and target of the path.
    # k is the amount of desired paths.
    # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.

    A = [nx.dijkstra_path(G, source, target, weight='weight')]
    A_len = [sum([G[A[0][l]][A[0][l + 1]]['weight'] for l in range(len(A[0]) - 1)])]
    B = []

    for i in range(1, k):
        for j in range(0, len(A[-1]) - 1):
            Gcopy = cp.deepcopy(G)
            spurnode = A[-1][j]
            rootpath = A[-1][:j + 1]
            for path in A:
                if rootpath == path[0:j + 1]:  # and len(path) > j?
                    if Gcopy.has_edge(path[j], path[j + 1]):
                        Gcopy.remove_edge(path[j], path[j + 1])
                    if Gcopy.has_edge(path[j + 1], path[j]):
                        Gcopy.remove_edge(path[j + 1], path[j])
            for n in rootpath:
                if n != spurnode:
                    Gcopy.remove_node(n)
            try:
                spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
                totalpath = rootpath + spurpath[1:]
                if totalpath not in B:
                    B += [totalpath]
            except nx.NetworkXNoPath:
                continue
        if len(B) == 0:
            break
        lenB = [sum([G[path[l]][path[l + 1]]['weight'] for l in range(len(path) - 1)]) for path in B]
        B = [p for _, p in sorted(zip(lenB, B))]
        A.append(B[0])
        A_len.append(sorted(lenB)[0])
        B.remove(B[0])

    return A, A_len


def get_laplacian_matrix(graph):
    '''
    返回图的拉普拉斯矩阵
    :param graph:
    :return:
    '''
    sup = nx.normalized_laplacian_matrix(graph).todense()
    return sup

def get_support_matrix(graph):
    pass

def nodeGraph_to_edgeGraph(graph, support=False):
    num_edges = graph.number_of_edges()
    edges_attr = {}
    idx = 0
    tran_dict = {'core layer': 1, 'converge layer': 2, 'access layer': 3, 'metro cross': 4}
    for u, v in graph.edges:
        attr = graph.get_edge_data(u, v)
        edges_attr[idx] = {'u': u, 'v': v, 'layer': tran_dict[attr['layer']], 'bandwidth': attr['bandwidth']}
        idx += 1
    edges_df = pd.DataFrame.from_dict(edges_attr)

    adj = np.zeros([num_edges, num_edges], dtype=np.int)
    for i in range(num_edges - 1):
        for j in range(i + 1, num_edges):
            if edges_attr[i]['u'] == edges_attr[i]['u'] \
                    or edges_attr[i]['v'] == edges_attr[i]['v'] \
                    or edges_attr[i]['u'] == edges_attr[i]['v'] \
                    or edges_attr[i]['v'] == edges_attr[i]['u']:
                adj[i, j] = 1
                adj[j, i] = 1
    sup = adj
    if support:
        edge_graph = nx.from_numpy_matrix(adj)
        sup = nx.normalized_laplacian_matrix(edge_graph).todense()
    f = edges_df.loc[['layer', 'bandwidth']]
    f = f.to_numpy()

    return sup, f[0, :], f[1, :]


def random_graph_input(num_nodes, p=0.3, support=True):
    graph = nx.erdos_renyi_graph(n=num_nodes, p=p)
    if not nx.is_connected(graph):
        c = list(nx.connected_components(graph))
        for j in range(len(c) - 1):
            graph.add_edge(list(c[j])[0], list(c[j + 1])[0])
    num_edges = graph.number_of_edges()
    edges_attr = {}
    idx = 0
    for u, v in graph.edges:
        edges_attr[idx] = {'u': u, 'v': v}
        idx += 1
    adj = np.zeros([num_edges, num_edges], dtype=np.int)
    for i in range(num_edges - 1):
        for j in range(i + 1, num_edges):
            if edges_attr[i]['u'] == edges_attr[i]['u'] \
                    or edges_attr[i]['v'] == edges_attr[i]['v'] \
                    or edges_attr[i]['u'] == edges_attr[i]['v'] \
                    or edges_attr[i]['v'] == edges_attr[i]['u']:
                adj[i, j] = 1
                adj[j, i] = 1
    edge_graph = nx.from_numpy_matrix(adj)
    sup = nx.normalized_laplacian_matrix(edge_graph).todense()
    bandwidth = np.ones([num_edges], dtype=np.float) * 1000

    return edge_graph, sup, bandwidth


def gen_flows(graph, size_percent: float(0 - 1.0) = 0.1, bandwidth=1000):
    source_nodes = list(graph.nodes)
    dest_nodes = list(graph.nodes)
    while True:
        s = int(np.random.choice(source_nodes, 1))
        d = int(np.random.choice(dest_nodes, 1))
        if s != d:
            yield [s, d, int(np.random.uniform(0.0e1 * bandwidth, size_percent * bandwidth))]


def gen_flows_zte(topo, num_flows, percent=0.05):
    flows = []
    flow_gen = topo.gen_flows(percent)
    for i in range(num_flows):
        flows.append(next(flow_gen))
    return flows


def cal_path_delay(path, now_cap, init_cap):
    delay = 0
    load = init_cap - now_cap
    for v in path:
        l = load[v]
        c = init_cap[v]
        d = 0
        if l / c <= 1 / 3:
            d = l
        elif 1 / 3 < l / c <= 2 / 3:
            d = 3 * l - 2 / 3 * c
        elif 2 / 3 < l / c <= 9 / 10:
            d = 10 * l - 16 / 3 * c
        else:
            d = 70 * l - 178 / 3 * c
        delay = max(delay, d)
    return delay


def cal_total_delay(traffic, init_bd):
    edges = len(init_bd)
    delay = np.zeros_like(init_bd,dtype=int)
    for e in range(edges):
        l = traffic[e]
        c = init_bd[e]
        if l / c <= 1 / 3:
            d = l
        elif 1 / 3 < l / c <= 2 / 3:
            d = 3 * l - 2 / 3 * c
        elif 2 / 3 < l / c <= 9 / 10:
            d = 10 * l - 16 / 3 * c
        elif 9 / 10 < l / c <= 1:
            d = 70 * l - 178 / 3 * c
        elif 1 < l / c < 11 / 10:
            d = 500 * l - 1468 / 3 * c
        else:
            d = 5000 * l - 16318 / 3 * c
        delay[e] = d
    return delay

