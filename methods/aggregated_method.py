# -----------------------------------------------------------------
# aggregated_method.py
#
# This file contains some utils for the experiment.
# November 5, Jianming Huang
# -----------------------------------------------------------------
import abc
import os
import json
import numpy as np
from itertools import product

# Method class for experiment
class exMethod:
    name = 'method'
    param = {} # Parameter dict of the method
    isKernel = False # If it is a kernel method
    useKernelFunc = False # If True, we will first compute the lambda kernel function on the gram matrix before svm

    @abc.abstractmethod
    def compute_X(self,G):
        # Function to compute the final feature data, that will be inputted in classifier
        return

class exTask:
    def __init__(self,method,param,dataset):
        # "exTask" takes 3 inputs: the first one is an instance of your method.
        # The second one is a dict of the parameters, note that if some of the items there are lists,
        # it will automatically run all the combinations of the parameters.
        # The third one is a list of the names of datasets, we here use the "tu_dataset" implemented by torch_geometric,
        # so please check you have entered the correct dataset names.
        self.method = method
        self.param = param
        self.dataset = dataset

    def generate_params(self):
        # Function to compute the combinations of parameters
        if not self.param:
            return [{}]
        ret = []
        loop_val = []
        for key in self.param.keys():
            p = self.param[key]
            if isinstance(p,list):
                loop_val.append(p)
            else:
                loop_val.append([p])
        for i in product(*loop_val):
            per_i = list(i)
            param = {}
            count = 0
            for key in self.param.keys():
                param[key] = per_i[count]
                count += 1
            ret.append(param)
        return ret