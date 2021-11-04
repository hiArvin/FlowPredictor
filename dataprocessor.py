from utils import *
import random
import time
from iteration_utilities import deepflatten
from label import delay_solver, solver
from topology import Topology


class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.random_bandwidth = args.random_bandwidth
        self.num_paths = args.num_paths
        self.num_flows = args.num_flows
        self.max_rate = args.max_rate
        self.min_rate = args.min_rate

        if 'ZTE' in args.graph_name:
            self.node_graph,self.topo = self.get_ZTE_graph(args.graph_name)
            self.generate_flows = self.generate_ZTE_flows
        else:
            self.node_graph = self.get_topozoo_graph(args.graph_name)
            self.generate_flows = self.generate_topozoo_flows
        for src,dst in nx.edges(self.node_graph):
            # print(src,dst)
            self.node_graph[src][dst]['weight']=1
        self.num_nodes = nx.number_of_nodes(self.node_graph)
        self.num_edges = nx.number_of_edges(self.node_graph)

        self.edge_graph = self.nodeGraph_2_edgeGraph()

        self.init_bandwidth()
        if self.num_nodes < 30:
            self.shortest_paths = self.gen_paths()
        else:
            self.shortest_paths = {}

    def get_ZTE_graph(self, graph_name):
        cfg = graph_name.split('-')
        num_core, num_converge, num_access = 1, 1, 1
        if len(cfg) == 4:
            _, num_core, num_converge, num_access = cfg
            num_core, num_converge, num_access = int(num_core), int(num_converge), int(num_access)
        topo = Topology(num_core=num_core, num_converge=num_converge, num_access=num_access)
        return topo.graph, topo

    def get_topozoo_graph(self, graph_name):
        dataset_path = "dataset/"
        graph_path = dataset_path + graph_name
        graph = nx.read_graphml(graph_path, node_type=int,force_multigraph=True)
        graph = nx.Graph(graph)
        # print(graph.edges)
        return graph

    def nodeGraph_2_edgeGraph(self):
        '''
        返回 edge graph 和 对应的边属性
        :param graph:
        :return:
        '''
        num_nodes = self.node_graph.number_of_nodes()
        num_edges = self.node_graph.number_of_edges()
        e_graph = nx.empty_graph(num_edges)
        edges = list(nx.edges(self.node_graph))
        for i in range(num_nodes):
            idx = []
            for nei in nx.neighbors(self.node_graph, i):
                if nei > i:
                    idx.append(edges.index((i, nei)))
                else:
                    idx.append(edges.index((nei, i)))
            for j in idx:
                for k in idx:
                    if j != k:
                        e_graph.add_edge(j, k)
        return e_graph

    def get_laplacian_matrix(self):
        sup = nx.normalized_laplacian_matrix(self.edge_graph).todense()
        return sup

    def init_bandwidth(self):
        if 'ZTE' in self.args.graph_name:
            bandwidth = np.zeros(self.num_edges)
            for i, edge in enumerate(self.topo.graph.edges.data()):
                bandwidth[i]=edge[2]['bandwidth']
        elif not self.random_bandwidth:
            bandwidth = np.ones(self.num_edges) * 10000
        else:
            bandwidth = np.random.randint(1, 10, self.num_edges) * 10000
        self.bandwidth = bandwidth

    def gen_paths(self):
        # 字典套字典形式的前n个最短路径，不足的用最短路径补齐
        shortest_path = {}
        for src in range(self.num_nodes):
            shortest_path[src] = {}
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                k_sp, k_len = k_shortest_paths(self.node_graph, source=src, target=dst, k=self.num_paths)
                for i in range(self.num_paths - len(k_sp)):
                    k_sp.append(k_sp[0])
                random.shuffle(k_sp)
                shortest_path[src][dst] = k_sp
        return shortest_path

    def gen_s_d_paths(self,src,dst,num_paths):
        k_sp,k_len = k_shortest_paths(self.node_graph, source=src, target=dst, k=num_paths)
        for i in range(num_paths - len(k_sp)):
            k_sp.append(k_sp[0])
        random.shuffle(k_sp)
        tmp = {dst: k_sp}
        if src in self.shortest_paths:
            self.shortest_paths[src].update(tmp)
        else:
            self.shortest_paths[src]= tmp
        # print(k_sp)
        return k_sp

    def generate_ZTE_flows(self):
        flows = []
        flow_generator = self.topo.gen_flows(self.max_rate,self.min_rate)
        for i in range(self.num_flows):
            flows.append(next(flow_generator))
        return flows

    def generate_topozoo_flows(self):
        # ZTE generator
        flows = []
        count = 0
        while (count < self.num_flows):
            nodes = list(self.node_graph.nodes)
            s, d = random.sample(nodes, 2)
            flow_size = np.random.randint(self.min_rate * self.bandwidth, self.max_rate * self.bandwidth, 1)
            if nx.shortest_path_length(self.node_graph, s, d) >= 5:
                flows.append([s, d, int(flow_size)])
                count += 1
        return flows

    def flow_to_numpy(self, flows):
        # print('shortest paths (nodes): ',shortest_path)
        sp = np.zeros([len(flows), self.num_paths, self.num_edges], dtype=np.int)
        edges = pd.DataFrame(nx.edges(self.node_graph, nbunch=None), columns=['src', 'dst'])
        for f in range(len(flows)):
            src, dst, flow_size = flows[f]
            if src in self.shortest_paths and dst in self.shortest_paths[src]:
                shortest_paths = self.shortest_paths[src][dst]
            else:
                shortest_paths = self.gen_s_d_paths(src,dst,self.num_paths)
            for p in range(self.num_paths):
                # print(src,dst,p)
                simple_path = shortest_paths[p]
                for i in range(len(simple_path) - 1):
                    idx = edges[((edges.src == simple_path[i]) & (edges.dst == simple_path[i + 1])) |
                                ((edges.dst == simple_path[i]) & (edges.src == simple_path[i + 1]))].index[0]
                    sp[f, p, idx] = flow_size
                # print( sp_numpy[f, p,:])
        # sp_numpy = sp_numpy.reshape([len(flows) * self.num_paths, self.num_edges])
        return sp

    def generate_seqs(self, flows):
        paths = []
        idx_list = []
        seqs = []
        edges = pd.DataFrame(nx.edges(self.node_graph, nbunch=None), columns=['src', 'dst'])
        for i in range(len(flows)):
            src, dst, flow_size = flows[i]
            for p in range(self.num_paths):
                pp = []
                path = self.shortest_paths[src][dst][p]
                for i in range(len(path) - 1):
                    idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                                ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
                    pp.append(idx)
                paths.append(pp)
        for i in range(len(paths)):
            seqs.append(list(range(len(paths[i]))))
            for j in range(len(paths[i])):
                idx_list.append(i)
        paths = list(deepflatten(paths))
        seqs = list(deepflatten(seqs))
        return paths, idx_list, seqs

    def generate_delay_label(self, sp, traffic, init_bd):
        res_selct, delay = delay_solver(self.num_edges, self.num_flows, self.num_paths, sp, traffic, init_bd)
        # convert to label
        label = np.zeros([self.num_flows, self.num_paths], dtype=np.int64)
        for q in range(self.num_flows):
            for p in range(self.num_paths):
                if res_selct[q, p] == 1:
                    label[q, p] = 1
                    break

        return label, delay

    def cal_delay_for_model(self, sp, outs):
        outs = np.eye(self.num_paths, dtype=int)[outs]
        traffic = np.sum(np.multiply(np.expand_dims(outs, -1).repeat(self.num_edges, -1), sp), axis=1)
        traffic = np.sum(traffic, axis=0)
        # print(traffic.shape)
        delay = cal_total_delay(traffic, self.bandwidth)
        return delay

    def sequential_delay_outputs(self, flows):
        bandwidth = self.bandwidth
        traffic = np.zeros_like(bandwidth)
        outputs = np.zeros(len(flows), dtype=int)
        process_time =0
        delay = None
        for i in range(self.num_flows):
            flow = flows[i]
            sp = self.flow_to_numpy([flow])
            time_start = time.time()
            label, delay = delay_solver(self.num_edges, 1, self.num_paths, sp, traffic, self.bandwidth)
            time_end = time.time()
            process_time += (time_end-time_start)
            res = np.argmax(label[0])
            outputs[i] = res
            traffic = traffic + sp[0][res]
        return outputs, delay ,process_time
        ## 自己写的版本，速度好像更慢了。。。
        # delay_p = np.zeros([self.num_paths,self.num_edges])
        # tfc = np.zeros([self.num_paths,self.num_edges])
        # for i in range(self.num_flows):
        #     for p in range(self.num_paths):
        #         flow = flows[i]
        #         sp = self.flow_to_numpy([flow])[0]
        #         tfc[p] = traffic+sp[p]
        #         delay_p[p] = cal_total_delay(np.squeeze(tfc[p]),bandwidth)
        #     delay_sum = np.nansum(delay_p,axis=1)
        #     chosen = np.nanargmax(delay_sum)
        #     outputs[i] = chosen
        #     traffic = tfc[chosen]
        # total_delay = delay_p[outputs[-1]]
        # return outputs, total_delay

    def shortest_path_delay_outputs(self, flows):
        bandwidth = self.bandwidth
        traffic = np.zeros_like(bandwidth)
        outputs = np.zeros(len(flows), dtype=int)
        for i in range(len(flows)):
            sp = self.flow_to_numpy([flows[i]])[0]
            chosen = np.nanargmax(np.nansum(sp,axis=-1))
            outputs[i] = chosen
            traffic += sp[chosen]
        delay = cal_total_delay(traffic, bandwidth)
        return outputs, delay



if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.graph_name = "Aarnet.graphml"
            self.random_bandwidth = False
            self.num_paths = 5
            self.num_flows = 10
            self.max_rate = 0.05
            self.min_rate = 0.001


    args = Args()
    dp = DataProcessor(args)
    print(dp.bandwidth)
    # bandwidth = np.zeros(dp.num_edges)
    # count=0
    # for i,edge in enumerate(dp.topo.graph.edges.data()):
    #     # print(edge[2]['bandwidth'])
    #     bandwidth[i]=edge[2]['bandwidth']
    # print(dp.topo.graph.edges(data=True))
    # flows = dp.generate_flows()
    # print(flows)
    # outs_seq, delay_seq = dp.sequential_delay_outputs(flows)
    # sp_numpy = dp.flow_to_numpy(flows)
    # traffic = np.zeros_like(dp.bandwidth)
    # outs_gb, delay_gb = dp.generate_delay_label(sp_numpy, traffic, dp.bandwidth)
    # print(outs_seq)
    # print(np.argmax(outs_gb, axis=1))
    # print(delay_gb)
    # print(delay_seq)
    # print(sum(delay_seq) - sum(delay_gb))
    # outs = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    # delay_pd = dp.cal_delay_for_model(sp_numpy, outs)
    # print(delay_pd)
