# LCS Graph Kernel
This is a repository for the source code of our paper _LCS Graph Kernel Based on Wasserstein Distance in Longest Common Subsequence Metric Space_
(Journal of Signal Processing, 2021), which is available [here](https://arxiv.org/abs/2012.03612).

As decribed in our paper, we implement an LCS graph kernel which is based on the [longest common subsequence](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)
 (LCS) similarity of shortest path sequences. We also apply several strategies to reduce the computational complexity. If you are intereseted in the details, please check them 
out in our paper.
 
Note that the original implementation in our paper (LCS implementation) does not support graphs with continuous node/edge attributes, because of the limitation of the LCS. But a 
good news is that we implemented a new version (DTW implementation) by using the [dynamic time warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) (DTW), which make it 
support the continuous attributes well. This repository includes both of the LCS and the DTW implementations.

If you have any question, please feel free to contact me. <koukenmei@toki.waseda.jp>

## Dependencies
The code is successfully tested by using Python 3.7 on Windows 10. It relies on the following dependencies:

- numpy (>=1.18.1)
- scipy (>=1.4.1)
- scikit-learn (>=0.22.1)
- pylcs (>=0.06)
- tqdm (>=4.42.1)
- POT (>=0.7.0)
- dtw-python (>=1.1.4)
- torch-geometric (>=1.6.3)

## Run the experiment
In the simplest instance, you can run the experiment script 'main.py' as follows:
```bash
python3 main.py
```
which will run the FLCS kernel (LCS implementation) on the MUTAG dataset as default.

To manage the experimental settings, we implement a class "exTask", which is in a list variable "tasklist" (you can check in the "main.py").
 Here are several examples to change the settings:
 ```bash
exTask(all_methods['LCSkernel'],
{'s':[0.5],'rho':[0.2],'ot_maxIter':50,'ot_epsilon':0.01,'dist_mode':'euclidean','method':'LCS'},
['PTC_MR'])
```
which runs the FLCS kernel (LCS implementation) on the PTC_MR dataset (about the meaning of parameters, we explain them in the "LCSkernel.py").
 ```bash
exTask(all_methods['LCSkernel'],
{'s':[0.2, 0.5],'rho':[0.2, 0],'ot_maxIter':50,'ot_epsilon':0.01,'dist_mode':'euclidean','method':'LCS'},
['MUTAG','PTC_MR'])
```
which runs the FLCS kernel (LCS implementation) with 4 combinations of parameters (s = 0.2, rho = 0.2 and s = 0.2, rho = 0 and s = 0.5, rho = 0.2 and s = 0.5, rho = 0) 
on both of the MUTAG and the PTC_MR datasets.

To run the DTW implementation, you can change the value of "method" item in the parameter dict to a string 'DTW'. Please note that for the DTW implementation, you should 
set the "dist_mode" item to specify a metric for computing distance of node/edge features. Here we use the Euclidean distance as default.

## Citation
If you find this useful, please cite it by using the following Bibtex citation:
 ```bash
@article{huang2021lcs,
  title={LCS graph kernel based on Wasserstein distance in longest common subsequence metric space},
  author={Huang, Jianming and Fang, Zhongxi and Kasai, Hiroyuki},
  journal={Signal Processing},
  volume={189},
  pages={108281},
  year={2021},
  publisher={Elsevier}
}
```
