def confidence(itemsets, a, b):
    """
    Compute the confidence of the rule a -> b.
    """
    return __get_support_value__(itemsets, {a, b}) / __get_support_value__(itemsets, {a})


def lift(itemsets, a, b):
    """
    Compute the lift of the rule a -> b.
    """
    return confidence(itemsets, a, b) / __get_support_value__(itemsets, {b})


def __get_support_value__(itemsets, item):
    return itemsets['support'].loc[itemsets['itemsets'] == item].values[0]
