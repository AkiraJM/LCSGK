# -----------------------------------------------------------------
# main.py
#
# This file contains the code of the experiemnt of nested cv.
# November 5, Jianming Huang
# -----------------------------------------------------------------
import time

from methods.aggregated_method import *
from kernel_evaluation import linear_svm_evaluation
from kernel_evaluation import kernel_svm_evaluation_nested

import methods.LCSkernel as LCS

from torch_geometric.datasets.tu_dataset import TUDataset

# Methods for experiment
# Note that if you want to add a new method, you should implement a class inherited
# from "exMethod" which you can find in "methods.aggregated_method"
all_methods = {
    'LCSkernel':LCS.LCSkernel(),
}

tasklist = [
    # The list of tasks.
    # "exTask" takes 3 inputs: the first one is an instance of your method.
    # The second one is a dict of the parameters, note that if some of the items there are lists,
    # it will automatically run all the combinations of the parameters.
    # The third one is a list of the names of datasets, we here use the "tu_dataset" implemented by torch_geometric,
    # so please check you have entered the correct dataset names.

    exTask(all_methods['LCSkernel'],{'s':[0.2],'rho':[0.2],'ot_maxIter':50,'ot_epsilon':0.01,'dist_mode':'euclidean','method':'LCS'},
           ['MUTAG']),

    # exTask(all_methods['LCSkernel'],{'s':[0.5],'rho':[0.2],'ot_maxIter':50,'ot_epsilon':0.01,'dist_mode':'euclidean','method':'LCS'},
    #        ['PTC_MR']),
    #
    # exTask(all_methods['LCSkernel'],{'s':[0.5],'rho':[0.2],'ot_maxIter':50,'ot_epsilon':0.01,'dist_mode':'euclidean','method':'DTW'},
    #        ['BZR']),

]

random_state = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
SVC_C = np.logspace(-3, 3, num=7)
SVC_lambda = np.logspace(-4, 1, num=6)
np.random.seed(42)

results = []

for eachtask in tasklist:
    for eachdataset in eachtask.dataset:
        # --------------------------------------------------------------------------------------------------
        # LOAD DATASET
        # --------------------------------------------------------------------------------------------------
        if eachtask.method.name == 'LCSkernel' and eachtask.method.param['method'] == 'DTW':
            tud = TUDataset(root='./datasets/', name=eachdataset, use_node_attr=True, use_edge_attr=True)
        else:
            tud = TUDataset(root='./datasets/', name=eachdataset, use_node_attr=False, use_edge_attr=False)
        G = [_ for _ in tud]
        y = np.array([_.y[0].item() for _ in tud])

        all_matrices = []
        for eachpara in eachtask.generate_params():
            print('Method: ', eachtask.method.name,'dataset: ', eachdataset,'param: ',eachpara)
            eachtask.method.param = eachpara

            # --------------------------------------------------------------------------------------------------
            # COMPUTE THE GRAM MATRIX
            # --------------------------------------------------------------------------------------------------
            start = time.time()
            X = eachtask.method.compute_X(G)
            elapsed_time = time.time() - start

            if eachtask.method.useKernelFunc:
                for eachlamb in SVC_lambda:
                    all_matrices.append(np.exp(-eachlamb * np.array(X)))
            else:
                all_matrices.append(X)

        # --------------------------------------------------------------------------------------------------
        # NESTED CROSS-VALIDATION
        # --------------------------------------------------------------------------------------------------
        accs = []
        for eachseed in random_state:
            if eachtask.method.isKernel:
                all,complete = kernel_svm_evaluation_nested(all_matrices,y,num_repetitions=1,C=SVC_C,all_std=True,split_random_state=eachseed)
            else:
                all, complete = linear_svm_evaluation(all_matrices, y, num_repetitions=1, C=SVC_C, all_std=True,split_random_state=eachseed)
            for each in complete:
                accs.append(each)
                print(len(accs),'/',10*len(random_state),' : ',each)

        results.append(eachtask.method.name + " " + eachdataset + " " + str(eachpara) + " " + str(np.array(accs).mean()) + " ± " + str(np.array(accs).std()))
        for r in results:
            print(r)