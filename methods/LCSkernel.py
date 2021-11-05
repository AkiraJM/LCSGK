# -----------------------------------------------------------------
# LCSkernel.py
#
# This file contains the code of the implementaion of the LCS graph
# kernel.
# November 5, Jianming Huang
# -----------------------------------------------------------------
from tqdm import tqdm
import numpy as np
import math
import pylcs
import ot
from dtw import dtw
from methods.aggregated_method import *

from scipy.sparse.csgraph import shortest_path

import warnings

warnings.filterwarnings('ignore')

# scipy shortest path
def get_sp_matrices(d):
    return shortest_path(d,return_predecessors=True)

def get_path_s(start, goal, pred):
    return get_path_row(start, goal, pred[start])

def get_path_row(start, goal, pred_row):
    path = []
    i = goal
    while i != start and i >= 0:
        path.append(i)
        i = pred_row[i]
    if i < 0:
        return []
    path.append(i)
    return path[::-1]

def normalize(x):
    return 2/(1+np.exp(-x)) - 1

def calcDTW(seq1, seq2, node_dist, edge_dist = None):
    # Function to compute the DTW distance of two path sequences
    # seq1, seq2 - The objective sequences whose elements denote the indices of nodes and edges in two graphs.
    #              Note that for the node indices i, in the sequence it should be i; for the edge indices j,
    #              it should be -(j+1) in the sequence.
    # node_dist - The distance matrix of the node features of the two graphs that seq1, seq2 are belonging to respectively.
    # edge_dist - The distance matrix of the edge features of the two graphs that seq1, seq2 are belonging to respectively.

    n1 = len(seq1)
    n2 = len(seq2)
    cost = np.full((n1, n2), 1.0)

    if not edge_dist is None:
        nidx1 = np.arange(0, n1, 2)
        nidx2 = np.arange(0, n2, 2)
        xv, yv = np.meshgrid(nidx1, nidx2, sparse=False, indexing='ij')
        cost[xv.reshape(-1),yv.reshape(-1)] = normalize(node_dist[seq1[xv.reshape(-1)],seq2[yv.reshape(-1)]])
        eidx1 = np.arange(1, n1, 2)
        eidx2 = np.arange(1, n2, 2)
        xv, yv = np.meshgrid(eidx1, eidx2, sparse=False, indexing='ij')
        cost[xv.reshape(-1), yv.reshape(-1)] = normalize(edge_dist[np.abs(seq1[xv.reshape(-1)]) - 1, np.abs(seq2[yv.reshape(-1)]) - 1])
    else:
        nidx1 = np.arange(0, n1, 1)
        nidx2 = np.arange(0, n2, 1)
        xv, yv = np.meshgrid(nidx1, nidx2, sparse=False, indexing='ij')
        cost[xv.reshape(-1),yv.reshape(-1)] = normalize(node_dist[seq1[xv.reshape(-1)],seq2[yv.reshape(-1)]])

    ret = dtw(cost,distance_only=True)
    return ret.distance/(n1 + n2)

def generatePathSequence_DTW(edge_index, x, edge_attr = None, rho=0.5, s=0.5, dist_mode = 'hamming'):
    # Generate the path sequences (DTW implementation)
    # edge_index - Indices of connected nodes. Two dimensional array with size of 2 times the number of edges of the graph.
    # x - Feature vectors of nodes. Two dimensional array with size of the number of nodes times the number of feature dimensions.
    # edge_attr - Feature vectors of edges. None or Two dimensional array with size of the number of edges times the number of
    #             feature dimensions.
    # rho - The removal ratio, should be within [0, 1).
    # s - The merging radius, should be within [0, 1).
    # dist_mode - The name of metric for computing distance of feature vectors.

    num_edges = edge_index.shape[1]
    num_nodes = x.shape[0]

    A = np.full((num_nodes,num_nodes),np.inf)
    A[edge_index[0, :], edge_index[1, :]] = 1
    E = np.full((num_nodes,num_nodes),np.inf)
    E[edge_index[0, :], edge_index[1, :]] = np.arange(num_edges) + 1

    dis, path = get_sp_matrices(A)

    node_dist = ot.dist(x, x, metric=dist_mode)
    if not edge_attr is None:
        edge_dist = ot.dist(edge_attr,edge_attr, metric=dist_mode)
    else:
        edge_dist = None

    M = []
    cur_p = []
    maxd = np.max(np.ma.masked_invalid(dis))
    minv = math.floor(maxd * rho)

    for i in range(num_nodes):
        for j in range(num_nodes - i):
            if (dis[i][i + j] > minv and dis[i][i + j] != np.inf):
                apath = get_path_s(i,i+j,path)
                len_apath = len(apath)
                for index in range(len_apath):
                    cur_p.append(apath[index])
                    if (not edge_attr is None) and (index + 1 < len_apath):
                        eidx = int(E[apath[index],apath[index + 1]])
                        cur_p.append(-eidx)

                reversed_cur_p = [_ for _ in reversed(cur_p)]
                if len(M) == 0:
                    M.append([list(cur_p), 0])
                else:
                    min_cost = np.inf
                    indextmp = 0
                    for indexm in range(len(M)):
                        eachm = M[indexm][0]

                        cost = min(calcDTW(np.array(cur_p), np.array(eachm), node_dist, edge_dist),
                                     calcDTW(np.array(reversed_cur_p), np.array(eachm), node_dist, edge_dist))
                        if s > 0:
                            if cost < min_cost:
                                min_cost = cost
                                indextmp = indexm
                        else:
                            if cost > 0:
                                cost = np.inf
                            if cost < min_cost:
                                min_cost = cost
                                indextmp = indexm
                                break

                    if (min_cost < s):
                        objm = M[indextmp][0]
                        if len(cur_p) > len(objm):
                            M[indextmp][0] = list(cur_p)
                        M[indextmp][1] += 1
                    else:
                        M.append([list(cur_p), 0])
                cur_p.clear()
    return M

def LCSKernel_DTW(X_d, Y_d, node_dist, edge_dist, ot_maxIter = 50,ot_epsilon = 0.5):
    # Compute the LCS graph distance (DTW implemenation)
    # X_d, Y_d - The Path sequences of two graphs, they should be the outputs of function "generatePathSequence_DTW"
    # node_dist - The distance matrix of the node features of the two graphs that seq1, seq2 are belonging to respectively.
    # edge_dist - The distance matrix of the edge features of the two graphs that seq1, seq2 are belonging to respectively.
    # ot_maxIter - The maximum iterations of the Sinkhorn algorithm.
    # ot_epsilon - The value of epsilon of the Sinkhorn algorithm.

    N = len(X_d)
    M = len(Y_d)
    if N == 0 and M == 0:
        return 0
    if N == 0 or M == 0:
        return 1
    Cost = np.zeros((N, M))

    weight_x = []
    weight_y = []
    node1 = 0
    for v1 in X_d:
        node2 = 0
        weight_x.append(v1[1] + 1)
        apath1 = [_ for _ in v1[0]]
        reversed_apath1 = [_ for _ in reversed(v1[0])]
        for v2 in Y_d:
            if len(weight_y) < M:
                weight_y.append(v2[1] + 1)
            apath2 = [_ for _ in v2[0]]

            Cost[node1][node2] = min(calcDTW(np.array(apath1), np.array(apath2), node_dist, edge_dist),
                         calcDTW(np.array(reversed_apath1), np.array(apath2), node_dist, edge_dist))

            node2 = node2 + 1
        node1 = node1 + 1

    weight_x = np.array(weight_x)
    weight_y = np.array(weight_y)
    mass_x = 1.0 / np.sum(weight_x)
    mass_y = 1.0 / np.sum(weight_y)

    # Sinkhorn's algorithm
    a = np.ones((N))
    a[range(N)] = mass_x * weight_x[range(N)]
    b = np.ones((M))
    b[range(M)] = mass_y * weight_y[range(M)]

    P = ot.sinkhorn(a, b, Cost, ot_epsilon, numItermax=ot_maxIter)

    return np.sum(np.multiply(Cost, P))

def oneHotToLabel(x):
    return np.argmax(x)

def generatePathSequence_LCS(edge_index, x, edge_attr = None, rho=0.5, s=0.5):
    # Generate the path sequences (LCS implementation)
    # edge_index - Indices of connected nodes. Two dimensional array with size of 2 times the number of edges of the graph.
    # x - Feature vectors of nodes. Two dimensional array with size of the number of nodes times the number of feature dimensions.
    # edge_attr - Feature vectors of edges. None or Two dimensional array with size of the number of edges times the number of
    #             feature dimensions.
    # rho - The removal ratio, should be within [0, 1).
    # s - The merging radius, should be within [0, 1).

    num_edges = edge_index.shape[1]
    num_nodes = x.shape[0]

    A = np.full((num_nodes,num_nodes),np.inf)
    A[edge_index[0, :], edge_index[1, :]] = 1
    E = np.full((num_nodes,num_nodes),np.inf)
    E[edge_index[0, :], edge_index[1, :]] = np.arange(num_edges)

    dis, path = get_sp_matrices(A)

    M = []
    cur_p = []
    maxd = np.max(np.ma.masked_invalid(dis))
    minv = math.floor(maxd * rho)

    for i in range(num_nodes):
        for j in range(num_nodes - i):
            if (dis[i][i + j] > minv and dis[i][i + j] != np.inf):
                apath = get_path_s(i,i+j,path)
                len_apath = len(apath)
                for index in range(len_apath):
                    cur_p.append(oneHotToLabel(x[apath[index]]))
                    if (not edge_attr is None) and (index + 1 < len_apath):
                        cur_p.append(int(255-oneHotToLabel(edge_attr[int(E[apath[index]][apath[index+1]])])))

                str_p = "".join([chr(_) for _ in cur_p])
                reversed_str_p = "".join([chr(_) for _ in reversed(cur_p)])
                if len(M) == 0:
                    M.append([list(cur_p), 0])
                else:
                    min_cost = np.inf
                    indextmp = 0
                    for indexm in range(len(M)):
                        eachm = M[indexm][0]
                        str_m = "".join([chr(_) for _ in eachm])

                        lcslen = max(pylcs.lcs(str_p, str_m), pylcs.lcs(reversed_str_p, str_m))
                        maxlen = max(len(str_p), len(str_m))
                        cost = 1-lcslen / maxlen

                        if s > 0:
                            if cost < min_cost:
                                min_cost = cost
                                indextmp = indexm
                        else:
                            if cost > 0:
                                cost = np.inf
                            if cost < min_cost:
                                min_cost = cost
                                indextmp = indexm
                                break

                    if (min_cost < s):
                        objm = M[indextmp][0]
                        if len(cur_p) > len(objm):
                            M[indextmp][0] = list(cur_p)
                        M[indextmp][1] += 1
                    else:
                        M.append([list(cur_p), 0])
                cur_p.clear()
    return M

def LCSKernel_LCS(X_d, Y_d, ot_maxIter = 50,ot_epsilon = 0.5):
    # Compute the LCS graph distance (LCS implemenation)
    # X_d, Y_d - The Path sequences of two graphs, they should be the outputs of function "generatePathSequence_LCS"
    # ot_maxIter - The maximum iterations of the Sinkhorn algorithm.
    # ot_epsilon - The value of epsilon of the Sinkhorn algorithm.

    N = len(X_d)
    M = len(Y_d)
    if N == 0 and M == 0:
        return 0
    if N == 0 or M == 0:
        return 1
    Cost = np.zeros((N, M))

    weight_x = []
    weight_y = []
    node1 = 0
    for v1 in X_d:
        node2 = 0
        weight_x.append(v1[1] + 1)
        apath1 = "".join([chr(_) for _ in v1[0]])
        reversed_apath1 = "".join([chr(_) for _ in reversed(v1[0])])
        for v2 in Y_d:
            if len(weight_y) < M:
                weight_y.append(v2[1] + 1)
            apath2 = "".join([chr(_) for _ in v2[0]])

            lcslen = max(pylcs.lcs(apath1, apath2), pylcs.lcs(reversed_apath1, apath2))
            maxlen = max(len(apath1), len(apath2))
            Cost[node1][node2] = 1 - lcslen / maxlen

            node2 = node2 + 1
        node1 = node1 + 1

    weight_x = np.array(weight_x)
    weight_y = np.array(weight_y)
    mass_x = 1.0 / np.sum(weight_x)
    mass_y = 1.0 / np.sum(weight_y)

    # Sinkhorn's algorithm
    a = np.ones((N))
    a[range(N)] = mass_x * weight_x[range(N)]
    b = np.ones((M))
    b[range(M)] = mass_y * weight_y[range(M)]

    P = ot.sinkhorn(a, b, Cost, ot_epsilon, numItermax=ot_maxIter)

    return np.sum(np.multiply(Cost, P))

def compute_feature_vectors_DTW(graphs, rho, s, dist_mode):
    # Generate path sequences for all graphs (DTW implementation)

    bar = tqdm(total=len(graphs))
    bar.set_description('Shortest-path')
    path_sequences = []
    for eachdata in graphs:
        ps = generatePathSequence_DTW(eachdata.edge_index.cpu().detach().numpy(), eachdata.x.cpu().detach().numpy(),
                                      None if eachdata.edge_attr is None else eachdata.edge_attr.cpu().detach().numpy(),
                                      rho = rho, s=s, dist_mode = dist_mode)
        path_sequences.append(ps)
        bar.update(1)
    bar.close()
    return path_sequences

def compute_feature_vectors_LCS(graphs, rho, s):
    # Generate path sequences for all graphs (LCS implementation)

    bar = tqdm(total=len(graphs))
    bar.set_description('Shortest-path')
    path_sequences = []
    for eachdata in graphs:
        ps = generatePathSequence_LCS(eachdata.edge_index.cpu().detach().numpy(), eachdata.x.cpu().detach().numpy(),
                                      None if eachdata.edge_attr is None else eachdata.edge_attr.cpu().detach().numpy(),
                                      rho = rho, s=s)
        path_sequences.append(ps)
        bar.update(1)
    bar.close()
    return path_sequences

def compute_distances_LCS(path_sequences, ot_maxIter = 50, ot_epsilon = 0.5):
    # Compute the distance matrix for all graphs (LCS implementation)

    sizeofdata = len(path_sequences)
    lcs_distances = []
    bar = tqdm(total=len(path_sequences))
    bar.set_description('TRAINING')

    for graph_index_1 in range(sizeofdata):
        distances = []
        for graph_index_2 in range(sizeofdata):
            if graph_index_2 == graph_index_1:
                distances.append(0)
            elif (graph_index_2 < graph_index_1):
                distances.append(lcs_distances[graph_index_2][graph_index_1])
            else:
                distances.append(LCSKernel_LCS(path_sequences[graph_index_1], path_sequences[graph_index_2],
                                               ot_maxIter=ot_maxIter,ot_epsilon=ot_epsilon))
        lcs_distances.append(distances)
        bar.update(1)
    bar.close()
    return lcs_distances

def compute_distances_DTW(graphs, path_sequences, dist_mode, ot_maxIter = 50, ot_epsilon = 0.5):
    # Compute the distance matrix for all graphs (DTW implementation)

    sizeofdata = len(path_sequences)
    assert sizeofdata == len(graphs)
    lcs_distances = []
    bar = tqdm(total=len(path_sequences))
    bar.set_description('TRAINING')

    for graph_index_1 in range(sizeofdata):
        distances = []
        for graph_index_2 in range(sizeofdata):
            if graph_index_2 == graph_index_1:
                distances.append(0)
            elif (graph_index_2 < graph_index_1):
                distances.append(lcs_distances[graph_index_2][graph_index_1])
            else:
                node_dist = ot.dist(graphs[graph_index_1].x.cpu().detach().numpy(),
                                    graphs[graph_index_2].x.cpu().detach().numpy(), metric=dist_mode)
                if not graphs[0].edge_attr is None:
                    edge_dist = ot.dist(graphs[graph_index_1].edge_attr.cpu().detach().numpy(),
                                        graphs[graph_index_2].edge_attr.cpu().detach().numpy(), metric=dist_mode)
                else:
                    edge_dist = None
                distances.append(LCSKernel_DTW(path_sequences[graph_index_1], path_sequences[graph_index_2],
                                               node_dist, edge_dist, ot_maxIter=ot_maxIter,ot_epsilon=ot_epsilon))
        lcs_distances.append(distances)
        bar.update(1)
    bar.close()
    return lcs_distances

class LCSkernel(exMethod):
    def __init__(self):
        self.name = 'LCSkernel'
        # Parameters of the LCS graph kernel
        # s - The merging radius as described in the section 5.3 of our paper
        # rho - The removal ratio as described in the section 5.2 of our paper
        # ot_maxIter - The maximum iterations of the Sinkhorn algorithm.
        # ot_epsilon - The value of epsilon of the Sinkhorn algorithm.
        # dist_mode - The method for computing distances between feature vectors. We use the "ot.dist" function of package "POT" to compute distance.
        #             So for the name of metric, you could refer to the documents of "POT". This only works when you are using the DTW implementation.
        # method - The main implementation of the kernel. For now, there are two options:
        #          1) LCS - The original implementation that our paper used. The distance of path sequences is computed by using the LCS
        #          similarity of two sequences. Note that this is implemented by using C++ (package "pylcs"), so it will work faster.
        #          2) DTW - The one that we replace the LCS with the Dynamic Time Warping (DTW). It supports both of the discrete and continuous
        #          inputs. You should set the "dist_mode" when you are using this implementation.
        self.param = {'s':0.3,'rho':0.2,'ot_maxIter':50,'ot_epsilon':0.01,'dist_mode':'hamming','method':'DTW'}
        self.isKernel = True
        self.useKernelFunc = True

    def compute_feature_vectors(self, G):
        # Function for kernel method to compute feature vectors
        if self.param['method'] == 'DTW':
            return compute_feature_vectors_DTW(G,self.param['s'],self.param['rho'],self.param['dist_mode'])
        else:
            return compute_feature_vectors_LCS(G, self.param['s'], self.param['rho'])

    def compute_distances(self, G, X):
        # Function for kernel method to compute distances
        if self.param['method'] == 'DTW':
            return compute_distances_DTW(G, X, self.param['dist_mode'], self.param['ot_maxIter'],self.param['ot_epsilon'])
        else:
            return compute_distances_LCS(X, self.param['ot_maxIter'],self.param['ot_epsilon'])

    def compute_X(self, G):
        # Function to compute final feature data, which will be inputted in classifier
        path_sequences = self.compute_feature_vectors(G)
        lcs_distances = self.compute_distances(G,path_sequences)
        return lcs_distances