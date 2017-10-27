# 前言

Kaggle是一个数据分析建模的应用竞赛平台，有点类似KDD-CUP数据挖掘竞赛。企业或者研究者将问题背景、数据、期望指标等发布到Kaggle上，以竞赛的形式向广大的数据科学家征集解决方案。我们可以下载/分析数据，使用统计/机器学习/数据挖掘等知识，建立算法模型，得到结果并提交。

这次的项目来自Kaggle上比较热门的比赛，是关于泰坦尼克号灾难预测的机器学习项目。原地址如下：

- [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

# Overview

## Competition Description

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## Evaluation

### Goal

It is your job to predict if a passenger survived the sinking of the Titanic or not. 
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.

### Metric

Your score is the percentage of passengers you correctly predict. This is known simply as ["accuracy”](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification).

### Submission File Format

You should submit a csv file with exactly 418 entries **plus** a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:

- PassengerId (sorted in any order)
- Survived (contains your binary predictions: 1 for survived, 0 for deceased)

```java
PassengerId,Survived
 892,0
 893,1
 894,0
 Etc.
```

## Data

The data has been split into two groups:

- training set (train.csv)
- test set (test.csv)

**The training set **should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use [feature engineering ](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)to create new features.

**The test set **should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

### Data Dictionary

| **Variable** | **Definition**                           | **Key**                                  |
| ------------ | ---------------------------------------- | ---------------------------------------- |
| survival     | Survival                                 | 0 = No, 1 = Yes                          |
| pclass       | Ticket class                             | 1 = 1st, 2 = 2nd, 3 = 3rd                |
| sex          | Sex                                      |                                          |
| Age          | Age in years                             |                                          |
| sibsp        | # of siblings / spouses aboard the Titanic |                                          |
| parch        | # of parents / children aboard the Titanic |                                          |
| ticket       | Ticket number                            |                                          |
| fare         | Passenger fare                           |                                          |
| cabin        | Cabin number                             |                                          |
| embarked     | Port of Embarkation                      | C = Cherbourg, Q = Queenstown, S = Southampton |

### Variable Notes

**pclass**: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

# 说明

该项目的代码和实现说明在jupyter notebook中。