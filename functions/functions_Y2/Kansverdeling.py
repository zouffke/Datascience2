import matplotlib.pyplot as plt


def kansverdeling(data, xlabel: str, ylabel: str, title: str = "Kansverdeling"):
    fix, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar(data[0], data[1])
    ax.set_title(title)
    plt.show()
