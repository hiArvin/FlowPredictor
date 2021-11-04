import numpy as np
import time
import argparse
import tensorflow as tf
from utils import *
from models import PEM
import seaborn as sns
import matplotlib.pyplot as plt
from dataprocessor import DataProcessor


class Evaluate:
    def __init__(self, args, data_processor):
        self.args = args
        # hyper-parameters
        self.num_flows = args.num_flows
        self.num_paths = args.num_paths
        self.epochs = args.epochs
        self.model_path = args.model_path

        self.data_processor = data_processor
        self.num_edges = data_processor.num_edges
        self.placeholders = {
            'support': tf.placeholder(tf.float32, shape=(self.num_edges, self.num_edges)),
            'features': tf.placeholder(tf.float32),
            'labels': tf.placeholder(tf.int64, shape=(self.num_flows, self.num_paths)),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'paths': tf.placeholder(tf.int64),
            'index': tf.placeholder(tf.int64),
            'sequences': tf.placeholder(tf.int64)
        }
        self.model = PEM(num_paths=self.num_paths,
                         num_edges=self.num_edges,
                         num_quests=self.num_flows,
                         placeholders=self.placeholders,
                         gcn_input_dim=self.num_paths * self.num_flows,
                         gcn_hidden_dim=16,
                         gcn_output_dim=8,
                         pe_output_dim=4,
                         att_layers_num=3)
        self.sess = tf.Session()
        self.model.load(self.sess, self.model_path)

    def evaluate(self):
        # 0-opt, 1-pred, 2-seq, 3-sp
        delay_res = np.zeros([self.epochs,4])
        time_res = np.zeros([self.epochs,4])
        for epoch in range(self.epochs):
            bandwidth = self.data_processor.bandwidth
            flows = self.data_processor.generate_flows()

            sp = self.data_processor.flow_to_numpy(flows)
            paths, idx, seqs = self.data_processor.generate_seqs(flows)
            support_matrix = self.data_processor.get_laplacian_matrix()
            sp_flatten = sp.reshape([len(flows) * self.num_paths, self.num_edges])
            fp2 = sp_flatten / np.tile(bandwidth * self.args.max_rate, [self.num_flows * self.num_paths, 1])
            feed_dict = {
                self.placeholders['support']: support_matrix,
                self.placeholders['features']: fp2.T,
                self.placeholders['paths']: paths,
                self.placeholders['index']: idx,
                self.placeholders['sequences']: seqs,
            }
            time1 = time.time()
            outs_pd = self.sess.run(self.model.outputs, feed_dict=feed_dict)
            time2 = time.time()
            outs_pd = np.nanargmax(softmax(outs_pd), axis=1)
            delay_pd = self.data_processor.cal_delay_for_model(sp, outs_pd)
            outs_seq, delay_seq, seq_time= self.data_processor.sequential_delay_outputs(flows)
            traffic = np.zeros_like(bandwidth, dtype=int)
            time4 =time.time()
            label, delay_opt = self.data_processor.generate_delay_label(sp, traffic, bandwidth)
            time5 = time.time()
            outs_sp, delay_sp = self.data_processor.shortest_path_delay_outputs(flows)
            time6 = time.time()
            print("延迟", np.sum(delay_pd), np.sum(delay_seq), np.sum(delay_opt),np.sum(delay_sp))
            print("时间", time2 - time1, '\t', seq_time, '\t', time5 - time4,'\t',time6-time5)
            delay_res[epoch]=np.sum(delay_opt),np.sum(delay_pd),np.sum(delay_seq),np.sum(delay_sp)
            time_res[epoch]=time5 - time4, time2 - time1, seq_time,time6-time5
        np.save(self.args.model_path+'/delay_res.npy', delay_res)
        np.save(self.args.model_path+'/time_res.npy', time_res)

    def integrated_grads(self, M=20):
        bandwidth = self.data_processor.bandwidth
        flows = self.data_processor.generate_flows()
        sp = self.data_processor.flow_to_numpy(flows)
        ###
        mask = sp != 0
        mask=np.reshape(mask,[self.num_flows*self.num_paths,self.num_edges])
        print(sp)
        # mask = np.squeeze(mask)
        print(np.sum(mask, axis=0))
        ###
        paths, idx, seqs = self.data_processor.generate_seqs(flows)
        support_matrix = self.data_processor.get_laplacian_matrix()
        sp_flatten = sp.reshape([len(flows) * self.num_paths, self.num_edges])
        fp2 = sp_flatten / np.tile(bandwidth * self.args.max_rate, [self.num_flows * self.num_paths, 1])
        features = fp2.T
        ig = np.zeros([self.data_processor.num_edges, self.num_paths * self.num_flows])
        for m in range(M):
            feed_dict = {
                self.placeholders['support']: support_matrix,
                self.placeholders['features']: features * m / M,
                self.placeholders['paths']: paths,
                self.placeholders['index']: idx,
                self.placeholders['sequences']: seqs,
            }
            gradient = self.sess.run(self.model.cal_gradient(), feed_dict=feed_dict)
            ig += gradient[0]
        ig = features * ig / M
        return ig

    def visualize(self, g, if_end=False):
        # ax1, ax2 = plt.plot(figsize=(10, 5))
        sns.set(style='white')
        # g = g-np.min(g)/(np.max(g)-np.min(g))  # normalize
        if not if_end:
            plt.figure(figsize=(4, 4))
            sns.heatmap(g / np.max(g), cbar=False, cmap='YlGnBu', vmin=0, vmax=1, )
        else:
            sns.heatmap(g / np.max(g), cmap='YlGnBu', vmin=0, vmax=1, )

        plt.imshow(g)
        plt.axis("off")
        plt.savefig("visualization/test.png")
        plt.close()



    def att_matrix(self):
        bandwidth = self.data_processor.bandwidth
        flows = self.data_processor.generate_flows()
        sp = self.data_processor.flow_to_numpy(flows)
        paths, idx, seqs = self.data_processor.generate_seqs(flows)
        support_matrix = self.data_processor.get_laplacian_matrix()
        sp_flatten = sp.reshape([len(flows) * self.num_paths, self.num_edges])
        fp2 = sp_flatten / np.tile(bandwidth * self.args.max_rate, [self.num_flows * self.num_paths, 1])
        features = fp2.T
        self.feed_dict = {
            self.placeholders['support']: support_matrix,
            self.placeholders['features']: features,
            self.placeholders['paths']: paths,
            self.placeholders['index']: idx,
            self.placeholders['sequences']: seqs,
        }
        print(len(self.model.layers))
        outputs = self.sess.run(self.model.outputs,
                                feed_dict=self.feed_dict)
        print(outputs)
        # att = self.sess.run(self.model.outputs,self.model.layers[-3].attention_weights)
        # print(att)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_flows", type=int, help="Number of flows process in an epoch")
    parser.add_argument("--num_paths", type=int, help="Number of candidate paths for a flow")
    parser.add_argument("--epochs", type=int, help="Evaluation epochs")
    parser.add_argument("--model_path", help="Path of trained model")
    parser.add_argument("--max_rate", type=float, default=0.05, help="flow size / bandwidth")
    parser.add_argument("--min_rate", type=float, default=0.001, help="flow size / bandwidth")
    parser.add_argument("--random_bandwidth", default=False)
    parser.add_argument("--graph_name", default="Aarnet.graphml")

    args = parser.parse_args()
    dp = DataProcessor(args)
    eva = Evaluate(args, dp)
    eva.evaluate()
    # eva.att_matrix()
