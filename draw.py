# from utils import random_graph_input
# graph,support,f_bandwidth = random_graph_input(num_nodes=10)
# print(support)
from topology import Topology
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 0-opt, 1-pred, 2-seq, 3-sp
time_res = np.load('logs/1104-1503-Arpanet19728-F20-P5/time_res.npy')
print(time_res)
d = np.load('logs/1104-1503-Arpanet19728-F20-P5/delay_res.npy')
# data_df = pd.DataFrame(d.T)
# print(data_df)
# writer = pd.ExcelWriter('logs/test.xlsx')  #关键2，创建名称为hhh的excel表格
# data_df.to_excel(writer,'page_1',float_format='%.3f')
# writer.save()
d = np.average(d,axis=0)
opt= d[0]
d = d/opt
print(d)
x = ['Opt', 'Pred', 'Seq', 'SP']
#
plt.figure()
# ax1 = plt.subplot(121)
plt.title('Delay Ratio')

plt.bar(x,d,color=['r','g','b','c'])
plt.show()

# x_axis = list(range(d.shape[1]))
# # plt.plot(x_axis, d[0,:], color='red', label='Predictor delay')
# # plt.plot(x_axis, d[1,:], color='green', label='SP selection delay')
# plt.plot(x_axis, d[0,:]/d[2,:], color='red', label='Predictor/OPT Ratio')
# plt.plot(x_axis, d[1,:]/d[2,:], color='green', label='SP/OPT Ratio')

# plt.ylim((0.9,1.5))
# plt.plot(x_axis, d[1,:], color='green', label='SP selection Delay')

# plt.legend() # 显示图例
# plt.xlabel('Iteration times')
# plt.ylabel('Delay')
# ax2 = plt.subplot(122)
# # ax2.title('Number of Successful Transmitted Flow')
# ax2.plot(x_axis, s[0,:], color='red', label='Predictor successful number')
# ax2.plot(x_axis, s[1,:], color='green', label='SP selection successful number')
# plt.savefig('logs/comp.jpg')
# plt.show()
#
# class Draw(object):
#     def __init__(self, data_path):
#         self.data_path= data_path
#         self.delay_np = np.load(data_path+'/delay_res.npy')
#         self.time_np = np.load(data_path+'/time_res.npy')
#
#     def draw(self):
#         base_line = self.delay_np[2,:]

