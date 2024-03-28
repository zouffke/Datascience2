
def confidence(itemsets, a, b):
    """
    Compute the confidence of the rule a -> b.
    """
    return itemsets['support'].loc[itemsets['itemsets'] == {a, b}].values[0] / itemsets['support'].loc[itemsets['itemsets'] == {a}].values[0]
