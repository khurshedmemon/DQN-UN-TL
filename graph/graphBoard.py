import numpy as np
from collections import defaultdict
from graph.nbCentrality import NB_model
import random
import graph_tool.all as gt
import pickle
import pdb

class graphBoard:

    def __init__(self, g, config, terminal_round=10, nodes_to_probe=10, net_part_vis=1):
        self.g = g
        #make two graphs for each party        
        self.p1_g = ""
        self.p2_g = ""
        self.NB_model = NB_model(self.g)
        self.kmeans = self.g.vp.kmeans
        #self.k = self.NB_model.k
        #Global community
        self.k = 1
        #self.initialize()        
        self.step = config.step        
        self.terminal_round = terminal_round        
        self.net_part_vis = net_part_vis     # %of network partial visibility to each party at the first round
        self.fixed_ini_vis = config.fix_ini_st
        self.env_name = config.env_name
        self.tr_scale = config.scale
        self.is_train = config.is_train
        self.nodes_to_probe = nodes_to_probe
        self.times_to_probe_nodes = config.times_to_probe_nodes     #R times to probe m nodes
        #assume max degree of a node is (n-1), since max degree of a node can not exceed (n-1)
        self.ini_max_deg = (self.g.num_vertices() - 1)
        self.ini_max_weight = (self.g.num_vertices() - 1)
        self.ini_unprobed = self.g.num_vertices()
        self.rnd_lists_idx = []
        self.testing_episode = config.testing_episode
        self.eps_counter = 0

        #self.expectation = np.zeros((self.g.num_vertices(), 5 +  (self.step + 1) * 2))  #Probed_Status, Activatiuon_Status, Degree, Weight, Blocking, SubGreedy
        #self.expectation = np.zeros((self.g.num_vertices(), 5))  #Probed_Status, Activatiuon_Status, Degree, Weight, Blocking
        self.expectation = np.ones(5)
        self.initialize()
        #self.nbCentrality = self.NB_model.centrality
        self.df_column_names = ['Prob_St', 'Act_Status', 'Deg', 'Weig', 'Blk_Wei', 'SubGeedy_1', 'SubGeedy_2', 'SubGeedy_3', 'SubGeedy_4', 'SubGeedy_5', 'SubGeedy_6', 'SubGeedy_7', 'SubGeedy_8']        
        self.init_state = self.expectation        
        #self.action_space = {"probing": 0, "degree": 1, "weight": 2, "blocking": 3, "subGreedy": 4, "centrality": 5, "randomy": 6, "voting": 7}
        #self.action_space = {"probing": 0, "degree": 1, "weight": 2, "blocking": 3, "subGreedy": 4, "randomy": 5, "voting": 6}
        self.action_space = {"probing": 0, "degree": 1, "weight": 2, "blocking": 3, "subGreedy": 4, "voting": 5}
        self.action_space_size = len(self.action_space)        
        self.action_size = self.action_space_size * self.k

        self.reset()

    def initialize(self):
        edgeMap = {}
        free_nodes = []
        max_deg = 0
        max_weight = 0
        #initilize sub-graphs
        #fixed edge-weights 0.4
        #for e in self.g.edges():            
            #self.g.ep.weight[e] = 0.4
            #print('edge: ', e, 'edge-weight: ', self.g.ep.weight[e])
        '''
        for v in self.g.vertices():
            for w in v.out_neighbours():
                edgeMap.setdefault(int(v), []).append(self.g.edge(v, w))
        self.edgeMap = edgeMap
        '''
        '''
        for k in range(self.k):
            vp_com = self.g.new_vp("bool", self.g.vp.kmeans.a == k)
            self.g.set_vertex_filter(vp_com)
            free_nodes.append(list(self.g.vertices()))
            self.g.clear_filters()
        '''

        free_nodes = list(self.g.vertices())
        self.init_free_nodes = free_nodes
        for m in range(5):
            self.expectation[m]=3
        
        self.init_free_nodes = free_nodes
        self.init_visited = self.g.vp.visited.copy()
        self.init_thres = self.g.vp.thres.copy()
        self.init_probed = self.g.vp.probed.copy()
        #if random initial visibility then load the already saved random lists
        if not(self.fixed_ini_vis):
            self.random_lists = self.get_initial_random_lists(self.net_part_vis)            
            if (self.is_train):
                self.rnd_lists_idx = np.random.randint(5000, size=self.tr_scale)
            else:
                #self.rnd_lists_idx = np.random.randint(2000, size=self.testing_episode)
                self.rnd_lists_idx = [x for x in range(2000)]

    def get_initial_random_lists(self, net_perc_vis):
        net_perc_vis_int = int(net_perc_vis)
        net_perc_vis_flt = float(net_perc_vis)
        #wheter to load training random lists or testing/evaluation random lists
        if (self.is_train):
            rnd_lists_file_int = "../data/" + self.env_name + "_tr_rand_lists/" + str(net_perc_vis_int) + ".xml.gz"
            rnd_lists_file_flt = "../data/" + self.env_name + "_tr_rand_lists/" + str(net_perc_vis_flt) + ".xml.gz"
            try:            
                with open (rnd_lists_file_flt, 'rb') as fp:
                    rnd_lists = pickle.load(fp)
            except:            
                with open (rnd_lists_file_int, 'rb') as fp:
                    rnd_lists = pickle.load(fp)
        else:
            #load evaluation random lists
            rnd_lists_file_int = "../data/" + self.env_name + "_tst_rand_lists/" + str(net_perc_vis_int) + ".xml.gz"
            rnd_lists_file_flt = "../data/" + self.env_name + "_tst_rand_lists/" + str(net_perc_vis_flt) + ".xml.gz"
            try:            
                with open (rnd_lists_file_flt, 'rb') as fp:
                    rnd_lists = pickle.load(fp)
            except:            
                with open (rnd_lists_file_int, 'rb') as fp:
                    rnd_lists = pickle.load(fp)
        return rnd_lists

    def random_thres(self):
        self.g.vp.thres = self.g.new_vp("double", 0.0)
        for v in self.g.vertices():
            sample = round(np.random.normal(0.5, 0.125), 2)
            if not (sample <= 0 or sample >= 1):
                self.g.vp.thres[v] = sample

    def edge_maps(self):
        edg_mps = {}
        for v in self.g.vertices():
            for w in v.out_neighbours():
                edg_mps.setdefault(int(v), []).append(self.g.edge(v, w))        
        return edg_mps
        #self.edgeMap = edgeMap

    def reset(self):
        self.g.vp.visited = self.init_visited.copy()
        self.g.vp.thres = self.init_thres.copy()        
        #self.g.vp.thres = self.random_thres()
        #self.random_thres()
        self.g.vp.probed = self.init_probed.copy()
        self.g.vp.thres_p1 = self.g.new_vp("double", 0.0)
        self.g.vp.thres_p2 = self.g.new_vp("double", 0.0)
        #self.free_nodes = [list(nodes_in_com) for nodes_in_com in self.init_free_nodes]
        self.free_nodes = [list(self.init_free_nodes)]
        self.terminal_st = False
        self.spreaders = []
        self.player = 1
        self.valid_com = [True for _ in range(self.k)]
        self.state = self.init_state
        #added by Ali
        self.p1_free_nodes = []
        self.p2_free_nodes = []
        self.p1_explored_nodes = []
        self.p2_explored_nodes = []
        self.p1_unexplored = []
        self.p2_unexplored = []
        self.p1_unexplored_visited = []
        self.p2_unexplored_visited = []
        self.visited_nodes = []
        self.p1_g = gt.Graph()
        self.p2_g = gt.Graph()        
        self.p1_g.add_vertex(self.g.num_vertices())
        self.p2_g.add_vertex(self.g.num_vertices())
        #initial network vivisiblity
        self.init_net_vis = int((self.net_part_vis/100) * self.g.num_vertices())
        print('Percentage to probe: ', self.net_part_vis, 'Nodes to probe of x %: ', self.init_net_vis)
        #all nodes are free and unxplored at initial time-stamp
        self.p1_free_nodes.append(list(self.g.vertices()))
        self.p2_free_nodes.append(list(self.g.vertices()))
        self.p1_unexplored.append(list(self.g.vertices()))
        self.p2_unexplored.append(list(self.g.vertices()))        
        prbe_node = -1
        #if initial network visibility counts to less than or equal to zero nodes then just pick a single random node
        if self.init_net_vis <= 0:
            self.init_net_vis = 1
        ### ************ initialize fix initial state for training ***********************
        if (self.fixed_ini_vis):            
            for m in range(self.init_net_vis):
                prbe_node = self.g.vertex(m)
                #get neighbors
                for w in self.g.get_out_neighbors(prbe_node):                
                    #add the node in visited but unexplored list
                    if w not in self.p1_explored_nodes:
                        self.p1_unexplored_visited.append(w)
                    if w not in self.p2_explored_nodes:
                        self.p2_unexplored_visited.append(w)
                #remove the node from explored list
                self.p1_explored_nodes.append(prbe_node)
                self.p2_explored_nodes.append(prbe_node)
                #add probing status of first party
                self.g.vp.probed[prbe_node] = 1        
        else:
            ### ************ initial network visibility based on snowball plus random sampling ***********************
            ''' commented due to random lists selection from stored random lists
            for m in range(self.init_net_vis):
                prob_rnd_pb = random.random()
                if prob_rnd_pb <= 0.5:
                    #select the node which has been visited but not probed yet
                    if len(self.p1_unexplored_visited) > 0:
                        prbe_node = self.g.vertex(random.choice(self.p1_unexplored_visited))
                        self.p1_unexplored_visited.remove(prbe_node)
                    else:                    
                        prbe_node = self.g.vertex(random.choice(self.p1_unexplored[0]))                    
                        self.p1_unexplored[0].remove(prbe_node)
                else:
                    #prbe_node = random.choice(self.p1_unexplored[0])                
                    prbe_node = self.g.vertex(random.choice(self.p1_unexplored[0]))                
                    self.p1_unexplored[0].remove(prbe_node)
                prbe_node = self.g.vertex(prbe_node)            
                #for e in self.edgeMap[int(prbe_node)]:
                for w in self.g.get_out_neighbors(prbe_node):
                #for w in prbe_node.out_neighbours():
                    #edge_vis = self.g.edge(prbe_node, w)
                    #self.p1_g.add_edge(prbe_node, w)                
                    #add the node in visited but unexplored list
                    if w not in self.p1_explored_nodes:
                        self.p1_unexplored_visited.append(w)
                #remove the node from explored list
                self.p1_explored_nodes.append(prbe_node)
                #add probing status of first party
                self.g.vp.probed[prbe_node] = 1
            '''
            #uniformly select a single generated random list
            #ini_vis = np.random.randint(0, 10)
            try:
                p1_probed_nodes = self.random_lists[self.rnd_lists_idx[self.eps_counter]]
            except:
                ini_vis = np.random.randint(0, 10)
                p1_probed_nodes = self.random_lists[ini_vis]
            for node in p1_probed_nodes:
                prbe_node = self.g.vertex(node)
                self.p1_unexplored[0].remove(prbe_node)
                for w in prbe_node.out_neighbours():
                    if w not in self.p1_explored_nodes:
                        self.p1_unexplored_visited.append(w)
                #remove the node from explored list
                self.p1_explored_nodes.append(prbe_node)
                #add probing status of first party
                self.g.vp.probed[prbe_node] = 1            
            #second party graph initilization
            prbe_node = -1
            #construct the network for second party
            for net_vis in range(self.init_net_vis):
                prob_rnd_pb = random.random()            
                if prob_rnd_pb <= 0.5:
                    #select the node which has been visited but not probed yet
                    if len(self.p2_unexplored_visited) > 0:
                        prbe_node = self.g.vertex(random.choice(self.p2_unexplored_visited))
                        #prbe_node = self.g.vertex(prbe_node)                    
                        self.p2_unexplored_visited.remove(prbe_node)
                    else:
                        #prbe_node = random.choice(self.p1_unexplored[0])                    
                        prbe_node = self.g.vertex(random.choice(self.p2_unexplored[0]))                    
                        self.p2_unexplored[0].remove(prbe_node)

                else:
                    #prbe_node = random.choice(self.p1_unexplored[0])                
                    prbe_node = self.g.vertex(random.choice(self.p2_unexplored[0]))                
                    self.p2_unexplored[0].remove(prbe_node)
                prbe_node = self.g.vertex(prbe_node)            
                #for e in self.edgeMap[int(prbe_node)]:
                for w in self.g.get_out_neighbors(prbe_node):                
                    if w not in self.p2_explored_nodes:
                        self.p2_unexplored_visited.append(w)
                #remove the node from explored list            
                self.p2_explored_nodes.append(prbe_node)
                #add probing status by second party
                #self.g.vp.probed[prbe_node] = 2
        self.eps_counter +=1        
        #self.update_state()
        

    def _play(self, action=0, opponent=None):
        self.target_com = int(action / self.action_space_size if not opponent else random.sample(list(np.where(self.valid_com)[0]), 1)[0])        
        strategy = action % self.action_space_size if not opponent else self.action_space[opponent]
        assert self.valid_com[self.target_com]
        self.target_com = 0
        
        print('Player: ', self.player, 'Strategy', strategy)        
        #it will return 0, no nodes to activate , just probe the network
        if strategy == 0:
            ind = self.probing()
        elif strategy == 1:
            ind = self.degree()
        elif strategy == 2:
            ind = self.weight()
        elif strategy == 3:
            ind = self.blocking()
        elif strategy == 4:
            ind = self.subGreedy()        
        elif strategy == 5:
            ind = self.voting()
        elif strategy == 6:
            ind = self.randomy()
        elif strategy == 7:
            ind = self.random_node()
        else:
            print("Invalid strategy number")
            return -1
        if ind == -1:
            return -1        
        #print ("player: ", self.player, ", ind: ", ind)        
        p1_reward = 0
        p2_reward = 0
        if ind != None:
            for v in self.g.vertices():
                if self.g.vp.visited[v] == 1:
                    p1_reward += 1
                elif self.g.vp.visited[v] == 2:
                    p2_reward += 1
            if self.player == 1:                
                self.p1_free_nodes[0].remove(ind)
            else:                
                self.p2_free_nodes[0].remove(ind)

            self.g.vp.visited[ind] = self.player
            self.free_nodes[self.target_com].remove(ind)
            self.spreaders.append(ind)
            self.visited_nodes.append(ind)
            

    def propagate(self, spreaders=None, visited=None, thres_list=None, virtual=False):
        if not virtual:
            spreaders = self.spreaders
            visited = self.g.vp.visited.a
            thres_list = [self.g.vp.thres.a, self.g.vp.thres_p1.a, self.g.vp.thres_p2.a]        
        weight = self.g.ep.weight

        candidates = set()
        for v in spreaders:
            player = visited[int(v)]
            #for e in self.edgeMap[int(v)]:
            for e in v.out_edges():
                if not visited[int(e.target())]:
                    thres_list[player][int(e.target())] += weight[e]
                    candidates.add(e.target())
        spreaders = []
        for v in candidates:
            max_thres = -1
            winner_list = []
            winner = -1
            for ind, thres_ in enumerate(thres_list):                
                if thres_[int(v)] > max_thres:
                    max_thres = thres_[int(v)]                    
                    winner_list = [ind]
                    winner = ind
                elif thres_[int(v)] == max_thres:
                    if winner != 0:
                        winner_list.append(ind)
            visited[int(v)] = random.sample(winner_list, 1)[0]
            if visited[int(v)]:
                #check if node is already explored than it can spread to other neighbors
                #if v in self.p1_explored_nodes or v in self.p2_explored_nodes:
                spreaders.append(v)
                if not virtual:                    
                    #self.free_nodes[self.g.vp.kmeans[v]].remove(v)
                    self.free_nodes[0].remove(v)

        if virtual:
            return spreaders
        else:
            self.spreaders = spreaders

    #greedy propagate
    def greedy_propagate(self, spreaders=None, visited=None, thres_list=None, virtual=True):
        if virtual:
            temp_spreaders = self.spreaders
            temp_visited = self.g.vp.visited.a
            temp_thres_list = [self.g.vp.thres.a, self.g.vp.thres_p1.a, self.g.vp.thres_p2.a]        
        temp_weight = self.g.ep.weight
        
        candidates = set()
        for v in temp_spreaders:
            player = temp_visited[int(v)]
            #for e in self.edgeMap[int(v)]:
            for e in v.out_edges():
                if not temp_visited[int(e.target())]:
                    temp_thres_list[player][int(e.target())] += temp_weight[e]
                    candidates.add(e.target())
        temp_spreaders = []
        for v in candidates:
            max_thres = -1
            winner_list = []
            winner = -1
            for ind, thres_ in enumerate(temp_thres_list):                
                if thres_[int(v)] > max_thres:
                    max_thres = thres_[int(v)]                    
                    winner_list = [ind]
                    winner = ind
                elif thres_[int(v)] == max_thres:
                    if winner != 0:
                        winner_list.append(ind)
            temp_visited[int(v)] = random.sample(winner_list, 1)[0]
            if temp_visited[int(v)]:
                #check if node is already explored than it can spread to other neighbors
                #if v in self.p1_explored_nodes or v in self.p2_explored_nodes:
                temp_spreaders.append(v)                

        if virtual:
            return temp_spreaders       

    #get_code
    def get_code(self, level):
        if level > 0.4:
            return 3
        elif level > 0.1:
            return 2
        elif level > 0.001:
            return 1
        else:
            return 0
    
    def update_state(self):
        deg_weight_dat = {}        
        max_deg, max_weight = 0, 0
        num_prb_vertices = (self.g.vp.probed.a == 1).sum()
        num_unprobed = self.g.num_vertices() - num_prb_vertices
        num_free_nodes = len(self.free_nodes[0])
        
        for com in range(self.k):
            self.valid_com[com] = True if self.free_nodes[com] else False
        
        if sum(self.free_nodes, []) == 0:
            self.is_terminal = True
            self.terminal_st = True
        else:
            #print('State before update: ', self.state)
            #prbing status
            #self.state[:, 0] = self.g.vp.probed.a
            self.state[0] = self.get_code(num_unprobed/self.g.num_vertices())
            #status-array
            #self.state[:, 1] = self.g.vp.visited.a
            self.state[1] = self.get_code(num_free_nodes/self.g.num_vertices())
            #degree and weight array            
            deg_weight_dat, max_deg, max_weight, self.upd_deg_weight_st()
            ''' commented
            for ind in deg_weight_dat:
                self.state[ind][2] = deg_weight_dat[ind][0]
                self.state[ind][3] = deg_weight_dat[ind][1]
            '''
            #generate randomly
            '''
            deg_rn_dig = np.random.randint(3)
            self.state[2] = deg_rn_dig
            weight_rn = np.random.randint(3)
            self.state[3] = weight_rn
            blck_rnd_weight = np.random.randint(3)
            self.state[4] = blck_rnd_weight
            '''
            self.state[2] = self.get_code(max_deg/self.ini_max_deg)            
            self.state[3] = self.get_code(max_weight/self.ini_max_weight)
            '''
            if max_deg != 0:
                self.state[:, 2] = np.around(self.state[:, 2] / max_deg, 2)
            if max_weight != 0:
                self.state[:, 3] = np.around(self.state[:, 3] / max_weight, 2)
            '''

            #block-array
            block_stat, block_dat, max_blk_weight = self.upd_block_st()
            self.state[4] = self.get_code(max_blk_weight/self.ini_max_weight)
            
            '''            
            if (block_stat):
                for ind, val in block_dat.items():
                    self.state[ind][4] = val
                if max_blk_weight != 0:
                    self.state[:, 4] = np.around(self.state[:, 4] / max_blk_weight, 2)
            '''
            
            #sub-greedy
            #self.state[:, 5:13] = self.calc_expectation(self.step)
            #self.nbCentrality = self.NB_model.build_centrality(True)
            #self.state = np.hstack((self.nbCentrality, self.expectation))

    ##Update vertices degree and weight
    def upd_deg_weight_st(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        max_deg, max_weight = 0, 0
        ind = None
        visited_a = np.copy(self.g.vp.visited.a)
        deg_st = defaultdict(list)
        #Target community, 0=Global community
        self.target_com = 0
        #just update first party node activation status
        #print('Free nodes: ', self.free_nodes)
        #pdb.set_trace()
        for v in self.p1_explored_nodes:
            if v not in self.visited_nodes and v in self.free_nodes[self.target_com]:                
                key = int(v)
                out_deg = 0
                out_deg = v.out_degree()
                out_weight = v.out_degree(weight=self.g.ep.weight)
                deg_st[key].append(out_deg)
                deg_st[key].append(out_weight)
                if out_deg > max_deg:
                    max_deg = out_deg
                if out_weight > max_weight:
                    max_weight = out_weight

        #return updated degree status        
        return deg_st, max_deg, max_weight

    #update vertices blocking weight status
    def upd_block_st(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        block_st = {}
        fringeNodes = []
        max_weight = 0
        #Target community, 0=Global community
        self.target_com = 0
        for v in self.spreaders:
            if self.g.vp.visited[v] == self._other_player:
                for w in v.out_neighbours():
                    if not self.g.vp.visited[w]:
                        if(self.player == 1):
                            if w in self.p1_explored_nodes and w in self.free_nodes[self.target_com]:
                                fringeNodes.append(w)
                        else:
                            if w in self.p2_explored_nodes and w in self.free_nodes[self.target_com]:
                                fringeNodes.append(w)

        if not fringeNodes:
            return False, block_st, max_weight

        max_weight = 0
        ind = None
        for v in fringeNodes:
            weight = 0
            weight = v.out_degree(weight=self.g.ep.weight)
            ind = int(v)
            block_st[key].append(weight)
            if weight > max_weight:
                max_weight = weight
        
        return True, block_st, max_weight

    def _other_player(self):
        return 1 if self.player == 2 else 2

    def switch_player(self):
        self.player = 1 if self.player == 2 else 2
    ##added by Ali
    #This function use the Probing strategy to explore the network  
    def probing(self):
        ind = None
        #first party probing        
        if(self.player == 1):
            #Probe m nodes for R times
            for r in range(self.times_to_probe_nodes):
                nodes_probe = (self.nodes_to_probe if self.nodes_to_probe <= len (self.p1_unexplored[0]) else len (self.p1_unexplored[0]))            
                for m in range(nodes_probe):
                    prob_rnd_pb = random.random()                    
                    if prob_rnd_pb <= 0.5:
                        #select the node which has been visited but not probed yet
                        if len(self.p1_unexplored_visited) > 0:
                            prbe_node = self.g.vertex(random.choice(self.p1_unexplored_visited))
                            self.p1_unexplored_visited.remove(prbe_node)
                        else:                    
                            prbe_node = self.g.vertex(random.choice(self.p1_unexplored[0]))                            
                            self.p1_unexplored[0].remove(prbe_node)
                    else:                        
                        prbe_node = self.g.vertex(random.choice(self.p1_unexplored[0]))
                        self.p1_unexplored[0].remove(prbe_node)
                    prbe_node = self.g.vertex(prbe_node)                    
                    for w in self.g.get_out_neighbors(prbe_node):
                        self.p1_g.add_edge(prbe_node, w)                        
                        #add the node in visited but unexplored list
                        if w not in self.p1_explored_nodes:
                            self.p1_unexplored_visited.append(w)
                    #add the node in explored list
                    self.p1_explored_nodes.append(prbe_node)
                    self.g.vp.probed[prbe_node] = self.player
        else:
            #probe for second party based on second-party visited network
            #Probe m nodes for R times
            for r in range(self.times_to_probe_nodes):
                nodes_probe = (self.nodes_to_probe if self.nodes_to_probe <= len (self.p2_unexplored[0]) else len (self.p2_unexplored[0]))
                for net_vis in range(nodes_probe):
                    prob_rnd_pb = random.random()                
                    if prob_rnd_pb <= 0.5:
                        #select the node which has been visited but not probed yet
                        if len(self.p2_unexplored_visited) > 0:
                            prbe_node = self.g.vertex(random.choice(self.p2_unexplored_visited))                        
                            self.p2_unexplored_visited.remove(prbe_node)
                        else:                        
                            prbe_node = self.g.vertex(random.choice(self.p2_unexplored[0]))                    
                            self.p2_unexplored[0].remove(prbe_node)
                    else:
                        prbe_node = self.g.vertex(random.choice(self.p2_unexplored[0]))
                        #prbe_node = random.choice(self.p2_unexplored[0])
                        self.p2_unexplored[0].remove(prbe_node)
                    prbe_node = self.g.vertex(prbe_node)
                    #add neighbors of probed node in a network 
                    for w in self.g.get_out_neighbors(prbe_node):
                        self.p2_g.add_edge(prbe_node, w)                    
                        #add the node in visited but unexplored list
                        if w not in self.p2_explored_nodes:
                            self.p2_unexplored_visited.append(w)                
                    self.p2_explored_nodes.append(prbe_node)
                    #self.g.vp.probed[prbe_node] = self.player
        #return 
        return ind

    def degree(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        max_degree = -1
        ind = None
        visited_a = np.copy(self.g.vp.visited.a)
        #select the Degree based on party's partial visibility
        if(self.player == 1):
            for v in self.p1_explored_nodes:
            #for v in self.p1_g.vertices():
                if v not in self.visited_nodes and v in self.free_nodes[self.target_com]:
                    degree = v.out_degree()
                    if degree > max_degree:
                        max_degree = degree
                        ind = v
        else:
            for v in self.p2_explored_nodes:
            #for v in self.p2_g.vertices():
                if v not in self.visited_nodes and v in self.free_nodes[self.target_com]:
                    degree = v.out_degree()
                    if degree > max_degree:
                        max_degree = degree
                        ind = v

        self.g.clear_filters()
        return ind

    def weight(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)        
        max_weight = -1
        ind = None
        visited_a = np.copy(self.g.vp.visited.a)
        if(self.player == 1):
            for v in self.p1_explored_nodes:
            #for v in self.p1_g.vertices():
                if v not in self.visited_nodes and v in self.free_nodes[self.target_com]:
                    weight = v.out_degree(weight=self.g.ep.weight)
                    if weight > max_weight:
                        max_weight = weight
                        ind = v
        else:
            for v in self.p2_explored_nodes:
            #for v in self.p2_g.vertices():
                if v not in self.visited_nodes and v in self.free_nodes[self.target_com]:
                    weight = v.out_degree(weight=self.g.ep.weight)
                    if weight > max_weight:
                        max_weight = weight
                        ind = v
        self.g.clear_filters()
        return ind

    def blocking(self):
        fringeNodes = []
        for v in self.spreaders:
            if self.g.vp.visited[v] == self._other_player:
                for w in v.out_neighbours():                    
                    if not self.g.vp.visited[w]:
                        if(self.player == 1):
                            if w in self.p1_explored_nodes and w in self.free_nodes[self.target_com]:
                                fringeNodes.append(w)
                        else:
                            if w in self.p2_explored_nodes and w in self.free_nodes[self.target_com]:
                                fringeNodes.append(w)
                        #fringeNodes.append(v)

        if not fringeNodes:
            return self.weight()

        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)

        max_weight = -1
        ind = None
        for v in fringeNodes:
            weight = v.out_degree(weight=self.g.ep.weight)
            if weight > max_weight:
                max_weight = weight
                ind = v

        self.g.clear_filters()
        return ind

    def subGreedy(self):
        #free_nodes = self.free_nodes[self.target_com]
        weight = self.g.ep.weight
        max_exp = -1
        ind = None
        visited = np.copy(self.g.vp.visited.a)
        thres_list = [np.copy(self.g.vp.thres.a), np.copy(self.g.vp.thres_p1.a), np.copy(self.g.vp.thres_p2.a)]
        #free nodes are explored nodes of each party
        if self.player == 1:
            free_nodes_rev = self.p1_explored_nodes            
            free_nodes_rv = [node for node in free_nodes_rev if node not in self.visited_nodes]
            free_nodes_spr = [node for node in free_nodes_rv if node in self.free_nodes[self.target_com]]
        else:
            free_nodes_rev = self.p2_explored_nodes            
            free_nodes_rv = [node for node in free_nodes_rev if node not in self.visited_nodes]
            free_nodes_spr = [node for node in free_nodes_rv if node in self.free_nodes[self.target_com]]
        for spreader in free_nodes_spr:
            visited = np.copy(self.g.vp.visited.a)
            thres_list = [np.copy(self.g.vp.thres.a), np.copy(self.g.vp.thres_p1.a), np.copy(self.g.vp.thres_p2.a)]
            candidates = [spreader]
            exp = 0
            for step in range(2):
                temp_candidates = []
                for v in candidates:
                    #for e in self.edgeMap[int(v)]:
                    for e in v.out_edges():
                        w = int(e.target())
                        if not visited[w]:
                            thres_list[self.player][w] += weight[e]
                            if thres_list[0][w] < thres_list[self.player][w]:
                                visited[w] = self.player
                                temp_candidates.append(e.target())
                exp += len(temp_candidates)
                candidates = temp_candidates

            if exp > max_exp:
                max_exp = exp
                ind = spreader

        return ind

    def centrality(self):
        local_cen = self.NB_model.local_cen_com[self.target_com]
        max_cen = -1
        ind = None
        for v in self.free_nodes[self.target_com]:
            cen = local_cen[v]
            if cen > max_cen:
                max_cen = cen
                ind = v

        return ind

    def randomy(self):
        rn = random.randint(1, 4)
        if rn == 1:
            return self.degree()
        if rn == 2:
            return self.weight()
        if rn == 3:
            return self.blocking()
        if rn == 4:
            return self.subGreedy()
        #if rn == 5:
            #return self.centrality()

    #random node selection
    def random_node(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        #max_degree = -1
        ind = None
        visited_a = np.copy(self.g.vp.visited.a)
        #select the random node based on party's partial visibility
        if self.player == 1:
            free_nodes_rev = self.p1_explored_nodes            
            free_nodes_rv = [node for node in free_nodes_rev if node not in self.visited_nodes]
            free_nodes_spr = [node for node in free_nodes_rv if node in self.free_nodes[self.target_com]]

        else:
            free_nodes_rev = self.p2_explored_nodes            
            free_nodes_rv = [node for node in free_nodes_rev if node not in self.visited_nodes]
            free_nodes_spr = [node for node in free_nodes_rv if node in self.free_nodes[self.target_com]]
        if not free_nodes_spr:
        #if len(free_nodes_spr) > 0:
            ind = random.choice(free_nodes_spr)        
        
        self.g.clear_filters()
        return ind

    def voting(self):
        ind = None
        '''
        candidates = [self.degree(),
                      self.weight(),
                      self.blocking(),
                      self.subGreedy(),
                      self.centrality()]
        '''
        candidates = [self.degree(),
                      self.weight(),
                      self.blocking(),
                      self.subGreedy()]        
        #if all are none then return ind
        if (all(x is None for x in candidates)):
            return ind
        else:
            unique, counts = np.unique(candidates, return_counts=True)            
            ind = np.argmax(counts)
            return unique[ind]

    def calc_expectation(self, step):  # player = 1 or 2
        spreaders = list(self.spreaders)
        visited_a = np.copy(self.g.vp.visited.a)
        thres_list = [np.copy(self.g.vp.thres.a), np.copy(self.g.vp.thres_p1.a), np.copy(self.g.vp.thres_p2.a)]
        exp_k_steps = [([np.equal(1, visited_a) * 1, np.equal(2, visited_a) * 1])]        
        
        for k in range(0, step):
            spreaders = self.greedy_propagate(spreaders, visited_a, thres_list, True)
            exp_k_steps.append([np.equal(1, visited_a) * 1, np.equal(2, visited_a) * 1])        
        
        exps = np.vstack((exp_k_steps[k] for k in range(step + 1)))        
        #exps = np.vstack((exp_k_steps[k] for k in range(1)))
        return exps.T

    def calc_reward(self):
        p1_reward = 0
        p2_reward = 0
        for v in self.g.vertices():
            if self.g.vp.visited[v] == 1:
                p1_reward += 1
            elif self.g.vp.visited[v] == 2:
                p2_reward += 1
        print ('P1 reward: ', p1_reward, '; P2 Reward: ', p2_reward)        
        #reward = p1_reward - p2_reward        
        reward = p1_reward
        return reward, p1_reward, p2_reward

    def is_terminal(self):
        #if len(self.p1_free_nodes[0]) == 0:
        '''
        if len(self.free_nodes) == 0: 
            return True
        else:
            return False
        '''
        if sum(self.free_nodes, []):
            return False
        else:
            return True

    def _terminal_state(self):
        return np.ones(self.state.shape)

    def __repr__(self):
        return 'test'