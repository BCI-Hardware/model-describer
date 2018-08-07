import pandas as pd

def prob_acc(true_class=0, pred_prob=0.2):
    """
    return classification prediction accuracy

    :param true_class: integer containing true label 0 or 1
    :param pred_prob: float - predicted probability
    :return scalar - prediction accuracy
    :rtype float
    """
    return (true_class * (1-pred_prob)) + ((1-true_class)*pred_prob)


def load_wine():
    """utility script to load wine dataset for examples"""

    wine = pd.read_csv('debug/wine.csv')
    return wine
