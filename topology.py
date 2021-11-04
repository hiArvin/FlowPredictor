import networkx as nx
import numpy as np


class Topology:
    def __init__(self, num_core: int = 1, num_converge: int = 2, num_access: int = 3, access_band: float = 1000):
        self.num_core = num_core
        self.num_converge = num_converge
        self.num_access = num_access
        self.access_band = access_band
        # ----在这个类中不可修改------
        self._num_node_core = 2 + 2
        self._num_node_converge = 4 + 2
        self._num_node_access = 20
        # -------------------------
        self.graph = nx.Graph()
        self._built_top()

    def _built_top(self):
        # 构建网络拓扑
        node_count = 0  # node序列计数
        for num_core in range(self.num_core):
            core_edges = [(x, y) for x in range(node_count, node_count + self._num_node_core - 1)
                          for y in range(x + 1, node_count + self._num_node_core)]  # 这里有很多冗余
            self.graph.add_edges_from(core_edges, layer='core layer', bandwidth=100 * self.access_band,
                                      IGP_num=num_core + 1,
                                      ring_num=1, delay=0.1)  # 核心层建立
            backbone_converge_nodes = [node_count + 1, node_count + 2]
            metro_core_nodes = [node_count, node_count + 3]
            node_count += self._num_node_core
            for num_converge in range(self.num_converge):
                converge_edges = [(x, x + 1) for x in range(node_count, node_count + self._num_node_converge - 1)]
                self.graph.add_edges_from(converge_edges, layer='converge layer', bandwidth=10 * self.access_band,
                                          IGP_num=num_core + 1, ring_num=num_converge + 1, delay=0.1)
                access_converge_nodes = [node_count + 2, node_count + 3]
                self.graph.add_edges_from(
                    list(zip(backbone_converge_nodes, [node_count, node_count + self._num_node_converge - 1])),
                    layer='converge layer', bandwidth=10 * self.access_band, IGP_num=num_core + 1,
                    ring_num=num_converge + 1, delay=0.1)
                node_count += self._num_node_converge
                for num_access in range(self.num_access):
                    access_edges = [(x, x + 1) for x in range(node_count, node_count + self._num_node_access - 1)]
                    self.graph.add_edges_from(access_edges, layer='access layer', bandwidth=self.access_band,
                                              IGP_num=num_core + 1, ring_num=num_access + 1, CONV_num=num_converge + 1,
                                              delay=0.1)
                    self.graph.add_edges_from(
                        list(zip(access_converge_nodes, [node_count, node_count + self._num_node_access - 1])),
                        layer='access layer', bandwidth=self.access_band, IGP_num=num_core + 1,
                        CONV_num=num_converge + 1, ring_num=num_access + 1, delay=0.1)
                    node_count += self._num_node_access
            if num_core > 0:
                metro_cross = list(zip(backbone_converge_nodes, metro_core_nodes_prev)) + list(
                    zip(backbone_converge_nodes_prev, metro_core_nodes))
                self.graph.add_edges_from(metro_cross, layer='metro cross', bandwidth=400 * self.access_band,
                                          IGP_num=num_core + 1, ring_num=0, delay=0.1)  # 连接不同的城域
            metro_core_nodes_prev = metro_core_nodes
            backbone_converge_nodes_prev = backbone_converge_nodes
        self._set_node_attr_()

    def _set_node_attr_(self):
        node_class = (
            'Metro Core', 'Backbone Convergence', 'Ordinary Convergence', 'Access Convergence', 'Access')
        for n, nbrs in self.graph.adjacency():
            layer_value = [self.graph[n][x]['layer'] for x in list(nbrs)]
            igp_value = [self.graph[n][x]['IGP_num'] for x in list(nbrs)]
            ring_value = [self.graph[n][x]['ring_num'] for x in list(nbrs)]
            self.graph.nodes[n]['IGP_num'] = min(igp_value)
            if all([x == 'access layer' for x in layer_value]):
                self.graph.nodes[n]['node class'] = node_class[4]
                self.graph.nodes[n]['layer'] = 'access layer'
                self.graph.nodes[n]['ring_num'] = ring_value[0]
                conv_value = [self.graph[n][x]['CONV_num'] for x in list(nbrs)]
                self.graph.nodes[n]['CONV_num'] = conv_value[0]
                continue
            elif all([x == 'converge layer' for x in layer_value]):
                self.graph.nodes[n]['node class'] = node_class[2]
                self.graph.nodes[n]['layer'] = 'converge layer'
                self.graph.nodes[n]['ring_num'] = ring_value[0]
                continue
            elif ('converge layer' in layer_value) and ('access layer' in layer_value):
                self.graph.nodes[n]['node class'] = node_class[3]
                self.graph.nodes[n]['layer'] = 'converge layer'
                self.graph.nodes[n]['ring_num'] = self.graph[n][n + 1]['ring_num']
                continue
            elif ('converge layer' in layer_value) and ('core layer' in layer_value):
                self.graph.nodes[n]['node class'] = node_class[1]
                self.graph.nodes[n]['layer'] = 'core layer'
                self.graph.nodes[n]['ring_num'] = self.graph.nodes[n]['IGP_num']
                continue
            else:
                self.graph.nodes[n]['node class'] = node_class[0]
                self.graph.nodes[n]['layer'] = 'core layer'
                self.graph.nodes[n]['ring_num'] = self.graph.nodes[n]['IGP_num']

    def gen_flows(self, max_rate=0.1, min_rate=0.01):
        source_nodes = []
        dest_nodes = list(self.graph.nodes)
        for node in self.graph.nodes:
            if self.graph.nodes[node]['layer'] == 'access layer':
                source_nodes.append(node)
        while True:
            s = int(np.random.choice(source_nodes, 1))
            d = int(np.random.choice(dest_nodes, 1))
            if self._helper_vail_s_d(s, d):
                yield [s, d, int(np.random.uniform(min_rate * self.access_band, max_rate * self.access_band))]

    def _helper_vail_s_d(self, s, d):
        s = self.graph.nodes[s]
        d = self.graph.nodes[d]
        if d['layer'] == 'core layer' or d['layer'] == 'metro cross':
            return True
        elif d['layer'] == 'converge layer':
            if s['IGP_num'] != d['IGP_num']:
                return True
            elif d['ring_num'] == s['CONV_num']:
                return False
        else:
            if s['IGP_num'] != d['IGP_num']:
                return True
            elif s['CONV_num'] != d['CONV_num']:
                return True
            elif s['ring_num'] != d['ring_num']:
                return True
            else:
                return False

    def edge_set(self):
        return np.array(list(self.graph.edges)).reshape((2, -1))


