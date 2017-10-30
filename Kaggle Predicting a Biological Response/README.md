# Predicting a Biological Response

这个Kaggle上关于由生物的分子信息判定是否会有生物反应的一个比赛，原地址如下:

- [Predicting a Biological Response](https://www.kaggle.com/c/bioresponse#description)

# Overview

## Description

The objective of the competition is to help us build as good a model as possible so that we can, as optimally as this data allows, relate molecular information, to an actual biological response.

We have shared the data in the comma separated values (CSV) format. Each row in this data set represents a molecule. The first column contains experimental data describing an actual biological response; the molecule was seen to elicit this response (1), or not (0). The remaining columns represent molecular descriptors (d1 through d1776), these are calculated properties that can capture some of the characteristics of the molecule - for example size, shape, or elemental constitution. The descriptor matrix has been normalized.

## Evaluation

Predicted probabilities that a molecule elicits a response are evaluated using the log loss metric.

Log loss is defined as:

# Data

The data is in the comma separated values (CSV) format. Each row in this data set represents a molecule. The first column contains experimental data describing a real biological response; the molecule was seen to elicit this response (1), or not (0). The remaining columns represent molecular descriptors (d1 through d1776), these are caclulated properties that can capture some of the characteristics of the molecule - for example size, shape, or elemental constitution. The descriptor matrix has been normalized.

- train.csv
- test.csv
- svm_benchmark.csv