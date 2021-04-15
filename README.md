# Awesome-Anomaly-Detection-Baseline

[TOC]

## Background

The repo reproduced baselines of the anomaly detection algorithm on the attribute graph.

## Requirements

- Python3
- torch==1.4.0
- scikit-learn==0.22.2.post1
- numpy==1.18.5
- networkx==1.11

## Return Values

+ **Recall@K**
+ **Precision@K**
+ **AUC value**

## Example Datasets

**Notes: **Before the experiments, you need to encapsulate your data in a ``.mat`` file. This ``.mat`` file includes three matrixes named ``A``, ``X``, ``gnd`` to represent adjacency matrix, attribute matrix and ground truth respectively. For example, in Amazon dataset, 1418 samples have 21 attributes for each one. The shape of ``A`` is ($1418\times1418$), the shape of ``X`` is ($1418\times21$) and the shape of ``gnd`` is ($1418\times1$).

|                 | Disney | Amazon | Enron   |
| --------------- | ------ | ------ | ------- |
| **#Nodes**      | 124    | 1,418  | 13,533  |
| **#Edges**      | 334    | 3,695  | 176,987 |
| **#Attributes** | 28     | 28     | 20      |

## Algorithms

+ **LOF:**[LOF: Identifying Density-Based Local Outliers](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1078.3580&rep=rep1&type=pdf)
+ **SCAN:**[A Structural Clustering Algorithm for Networks](https://dl.acm.org/doi/abs/10.1145/1281192.1281280)

+ **Radar:**[Radar: Residual Analysis for Anomaly Detection in Attributed Networks](https://www.ijcai.org/Proceedings/2017/299)
+ **ANOMALOUS:**[ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks](https://www.ijcai.org/Proceedings/2018/488)
+ **Dominant:**[Deep Anomaly Detection on Attributed Networks](http://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf)

## Usage

**Take Dominant Algorithm for example.**

```bash
cd Dominant
python -u "run.py"
```

## Welcome To Contribute

If you have any models implemented with great performance, you're welcome to contribute. Also, I'm glad to help if you have any problems with the project, feel free to raise an issue.

**Contact me:** xdu DOT lhchen AT gmail DOT com.
