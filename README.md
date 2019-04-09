# Efficient-SSL

This is a TensorFlow implementation of Improved Graph Convolutional Networks (IGCN) and Generalized Label Propagation (GLP) for the task of (semi-supervised) node classification, as described in our paper:

Li, Qimai, Xiao-Ming Wu, Han Liu, Xiaotong Zhang, and Zhichao Guan. ["Label Efficient Semi-Supervised Learning via Graph Filtering."](https://arxiv.org/abs/1901.09993) In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019.


## Requirements
* python3
* tensorflow (1.5.0 and 1.9.0 are tested)
* networkx
* numpy
* scipy

Anaconda environment is recommended.

## Run the demo

```bash
python train.py -v
```
## Reproduce Results in Paper
Reproduction may require average accuracy over multiple runs by adding option `--repeat 50`.

```bash
# citation network, 20 labels per class
python train.py --pset config_citation.large_label_set --dataset cora
python train.py --pset config_citation.large_label_set --dataset citeseer 
python train.py --pset config_citation.large_label_set --dataset pubmed
python train.py --pset config_citation.large_label_set --dataset large_cora --layer-size '[64]'

# citation network, 4 labels per class
python train.py --pset config_citation.small_label_set --dataset cora
python train.py --pset config_citation.small_label_set --dataset citeseer
python train.py --pset config_citation.small_label_set --dataset pubmed
python train.py --pset config_citation.small_label_set --dataset large_cora --layer-size '[64]'

# NELL
python train.py --pset config_nell.large_label_set --dataset nell.0.1
python train.py --pset config_nell.small_label_set --dataset nell.0.01
python train.py --pset config_nell.small_label_set --dataset nell.0.001
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).

Have a look at the `load_data()` function in `gcn/utils.py` for an example.

In this example, we load citation network data (Cora, CiteSeer, PubMed, Large Cora and NELL). The original datasets of Cora, CiteSeer, PubMed can be found here: https://linqs.soe.ucsc.edu/data. NELL's official sit is here: http://rtw.ml.cmu.edu/.

In our version (see `data` folder) Cora, CiteSeer, PubMed and NELL are split by [\[3\]](https://github.com/kimiyoung/planetoid).

Large Cora is constructed from [Andrew McCallum's Cora Project](https://people.cs.umass.edu/~mccallum/data.html), and processed by our own. If you use Large Cora dataset, please cite our paper "Label Efficient Semi-Supervised Learning via Graph Filtering" and McCallum's paper "Automating the Construction of Internet Portals with Machine Learning" 


## Models

You can choose between the following models: 
* `IGCN`: Improved Graph Convolutional Networks \[1\]
* `GLP`: Generalized Label Propagation \[1\]
* `GCN`: Graph convolutional network \[2\]
* `LP`: Label Propagation \[4\]
* `MLP`: Basic Multi-Layer Perceptron that supports sparse inputs.

## Cite

Please cite our paper if you use this code or Large Cora dataset in your own work:

```
@inproceedings{li2019label,
  title={Label Efficient Semi-Supervised Learning via Graph Filtering},
  author={Qimai Li and Xiao-Ming Wu and Han Liu and Xiaotong Zhang and Zhichao Guan},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2019},
  url={https://arxiv.org/abs/1901.09993},
}
```
## Acknowledgement
Thanks for [Kipf's implementation of GCN](https://github.com/tkipf/gcn/), on which this repository is initially based.

## References 
[1.] Q. Li, X.-M. Wu, H. Liu, X. Zhang, and Z. Guan. Label efficient semi-supervised learning via graph filtering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[2.] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In International Confer- ence on Learning Representations, 2017.

[3.] Z. Yang, W. W. Cohen, and R. Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. In International Conference on Machine Learning, pages 40–48, 2016.

[4.] X. Wu, Z. Li, A. M. So, J. Wright, and S.-f. Chang. Learning with Partially Absorbing Random Walks. In Conference on Neural Information Processing Systems, pages 3077–3085, 2012.