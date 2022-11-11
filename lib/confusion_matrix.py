from numpy import diag
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, Series, crosstab


def compare_test(path_to_save: str, test_classes, test_preds, inverted_map: dict) -> dict:
    """Calling for estimators and plotting of confusion matrix

    Args:
        test_classes (array): true classes we're dealing wih for our test set
        test_preds (array): predicted classes for test set
        inverted_map (dict): mapping between ints and class labels
        sample_name (str): used for naming purposes
        clade (str): level we're working at
        determined (str): upper level already determined

    Returns:
        dict: report with estimators
    """
    test_preds = [t for t in test_preds]
    cm = pandas_confusion(test_classes, test_preds,
                          inverted_map)
    plot_pandas(path_to_save, cm)


def pandas_confusion(test_classes, test_preds, inverted_map: dict) -> DataFrame:
    """Creates the dataframe used to plot a confusion matrix from test datas

    Args:
        test_classes (array): true classes we're dealing with for our test set
        test_preds (array): predicted classes for test set
        inverted_map (dict): mapping between ints and class labels

    Returns:
        pd.DataFrame: confusion matrix
    """
    test_classes = [inverted_map[str(int(t))] for i, t in enumerate(
        test_classes) if not isinstance(test_preds[i], bool)]
    test_preds = [inverted_map[str(int(t))]
                  for t in test_preds if not isinstance(t, bool)]
    data = {'y_Actual': test_classes, 'y_Predicted': test_preds}
    df = DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    return crosstab(df['y_Actual'], df['y_Predicted'], rownames=[
        'Actual'], colnames=['Predicted'])


def plot_pandas(path_to_save: str, cm: DataFrame, cmap: str = 'Purples') -> None:
    """Plots the confusion matrix for test data at given level

    Args:
        cm (pd.DataFrame): a pandas df that contains a confusion matrix
        sample_name (str): used for naming purposes
        clade (str): level we're working at
        determined (str): upper level already determined
        cmap (str, optional): set of colors for the heatmap. Defaults to 'bone'.
    """
    number_digits: int = 2
    plt.figure(figsize=(7, 6))
    ax = plt.axes()
    try:
        diag_axis = Series(diag(cm)).sum()
        full_set = cm.to_numpy().sum()
        ax.set_title(
            f"Confusion matrix - R={round(diag_axis/(full_set-diag_axis),2)}")
    except:
        ax.set_title(
            f"Confusion matrix")
    # percentage
    cm = cm.div(cm.sum(axis=1), axis=0) * 100
    sns.heatmap(cm, annot=True, cmap=cmap,
                fmt=f".{number_digits}f", linewidths=0.5, ax=ax)
    plt.savefig(
        f"{path_to_save}/conf_matrix.png", bbox_inches='tight')
