# Awesome-Anomaly-Detection-Baseline

[TOC]

## Background

The repo reproduced baselines of the anomaly detection algorithm on the attribute graph.

## Example Datasets

**Notes: **Before the experiments, you need to encapsulate your data in a ``.mat`` file. This ``.mat`` file includes three matrixes named ``A``, ``X``, ``gnd`` to represent adjacency matrix, attribute matrix and ground truth respectively. For example, in Amazon dataset, 1418 samples have 21 attributes for each one. The shape of ``A`` is ($1418\times1418$), the shape of ``X`` is ($1418\times21$) and the shape of ``gnd`` is ($1418\times1$).

|             | Disney | Books | Enron   |
| ----------- | ------ | ----- | ------- |
| #Nodes      | 124    | 1,418 | 13,533  |
| #Edges      | 334    | 3,695 | 176,987 |
| #Attributes | 28     | 28    | 20      |

## Algorithms

+ **Radar:**[Radar: Residual Analysis for Anomaly Detection in Attributed Networks](https://www.ijcai.org/Proceedings/2017/299)

+ **ANOMALOUS:**[ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks](https://www.ijcai.org/Proceedings/2018/488)

## Performance