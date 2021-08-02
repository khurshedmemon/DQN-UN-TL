from __future__ import print_function
import argparse
import os, sys
import numpy as np
import random
import tensorflow as tf
import graph_tool.all as gt
from dqn.agent import Agent
from dqn.network_env import NetEnv
from config import get_config
flags = tf.app.flags
import pdb

# Model
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', True, 'Whether to use double q-learning')
flags.DEFINE_string('mv', 'DRL_UN_rnd_st_mv_1_200k_WC_Prb10', 'Model version represents model training episodes, edge-weiths, and network visibility')
# Environment
flags.DEFINE_string('env_name', 'celegansneural','The name of graph to use')
flags.DEFINE_integer('scale', 200000,'Number of training episodes')
# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '2/3', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_string('gpu_id', '0', 'GPU Id to use')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('cnn_format', 'NCHW', 'INTERNAL USED ONLY')
#
flags.DEFINE_string('opponent', 'degree', 'oppoent strategy')
flags.DEFINE_integer('terminal_round', 10, 'Number of Rounds to explore and invest budget')
flags.DEFINE_float('net_part_vis', 1, 'initial network visibility at first round of training or testing')
flags.DEFINE_integer('nodes_to_probe', 10, 'Nodes to probe when probing strategy is taken')
flags.DEFINE_integer('times_to_probe_nodes', 1, 'R times to probe m number of nodes')
flags.DEFINE_boolean('fix_ini_st', False, 'Fixed initial network visibility i.e., same nodes probed at zero round')
flags.DEFINE_integer('prob_nodes_AI', 10, 'Number of nodes to probe when probing strategy is selected by first party')
flags.DEFINE_integer('prob_nodes_opp', 10, 'Number of nodes to probe when probing strategy is selected by second party')
flags.DEFINE_integer('AI_budget', 5, 'Budget by first party')
flags.DEFINE_integer('OP_budget', 5, 'Budget by second party')
flags.DEFINE_integer('testing_episode', 2000, 'Number of test episodes')
flags.DEFINE_boolean('use_tl', False, 'Whether to use pre-trained model or not (Transfer Learning)')
flags.DEFINE_string('tl_base_model', './checkpoints/celegansneural/trained_models/op_stra-degree/mv-DRL_UN_rnd_st_mv_1_200k_WC_Prb10/train_scale-200000/', 'pre-trained base model for further retraining, fine-tuning, or apply as-is')
flags.DEFINE_string('tl_bm_weights_dir', './checkpoints/celegansneural/weights/mv-DRL_UN_rnd_st_mv_1_200k_WC_Prb10/', 'baseline models weights stored in a directory')

FLAGS = flags.FLAGS
# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = idx / num
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def set_graph(g, config):
    print("Total num of vertices: %d" % g.num_vertices())
    print("Total num of edges: %d" % g.num_edges())

    g.set_directed(True)
    g.vp.visited = g.new_vp("int", 0)
    g.ep.weight = g.new_ep("double", 0.0)
    g.vp.probed = g.new_vp("int", 0)    #whether this node is probed by first party or second
    #g.vp.visited = g.new_vp("int", 0)
    '''
    for e in g.edges():
        g.ep.weight[e] = 1.0 / e.source().in_degree()
    '''
    #Revised edge-weights
    for v in g.vertices():
        for u in v.out_neighbours():
            #in_deg_u = u.in_degree()
            in_deg_u = 0
            for z in u.in_neighbours():
                in_deg_u += 1
            if(in_deg_u != 0):
                #expon_res = round(math.exp(-random.randint(1,10)*time_stamp), 2)
                g.ep.weight[g.edge(g.vertex_index[v], g.vertex_index[u])] = round((1/in_deg_u), 2)
            else:                
                g.ep.weight[g.edge(g.vertex_index[v], g.vertex_index[u])] = 1    

    g.vp.thres = g.new_vp("double", 0.0)
    g.vp.thres_p1 = g.new_vp("double", 0.0)
    g.vp.thres_p2 = g.new_vp("double", 0.0)

    for v in g.vertices():
        sample = round(np.random.normal(0.5, 0.125), 2)
        if not (sample <= 0 or sample >= 1):
            g.vp.thres[v] = sample
    '''
    #check the edge weights
    for e in g.edges():
        print('Edge e: ', e, 'Edge weight: ', g.ep.weight[e])
    '''
    g.save("../data/" + config.env_name + "_weighted.txt.xml.gz")


def main(_):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))    
    #which gpu to use
    with tf.device('/gpu:' + str(FLAGS.gpu_id)):        
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)) as sess:
            config = get_config(FLAGS) or FLAGS
            #set parameters            
            try:
                g = gt.load_graph("../data/" + config.env_name + "_weighted.txt.xml.gz")
            except:
                g = gt.load_graph("../data/" + config.env_name + ".txt.xml.gz")
                set_graph(g, config)
            print('Graph Stats: , Total Number of Nodes; ', g.num_vertices(), 'Total Number of edges: ', g.num_edges())            
            opponent = FLAGS.opponent
            terminal_round = FLAGS.terminal_round
            nodes_to_probe = FLAGS.nodes_to_probe
            net_part_vis = FLAGS.net_part_vis
            env = NetEnv(g, config, opponent, terminal_round, nodes_to_probe, net_part_vis)

            if not tf.test.is_gpu_available() and FLAGS.use_gpu:
                raise Exception("use_gpu flag is true when no GPUs are available")
            
            if not FLAGS.use_gpu:
                config.cnn_format = 'NHWC'

            agent = Agent(config, env, sess)
            if FLAGS.is_train:
                agent.train()            
            else:
                agent.play(test_ep=0.000001)

if __name__ == '__main__':
    tf.app.run()
