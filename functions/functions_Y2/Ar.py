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


def leverage(itemsets, a, b):
    """
    Compute the leverage of the rule a -> b.
    """
    return __get_support_value__(itemsets, {a, b}) - __get_support_value__(itemsets, {a}) * __get_support_value__(
        itemsets, {b})

def conviction(itemsets, a, b):
    """
    Compute the conviction of the rule a -> b.
    """
    return (1 - __get_support_value__(itemsets, {b})) / (1 - confidence(itemsets, a, b))


def __get_support_value__(itemsets, item):
    return itemsets['support'].loc[itemsets['itemsets'] == item].values[0]
