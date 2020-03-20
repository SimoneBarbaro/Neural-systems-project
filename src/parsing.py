import pandas as pd
import os


def parse_exemple_file(folder="."):
    """
    Load exemple file.
    :param folder: folder where file resides.
    :return: dataframe parsed from file.
    """
    return pd.read_csv(os.path.join(folder, "dataset.txt"), sep=r"<-SEPARATOR->",
                       header=0, index_col=False, dtype="str",  keep_default_na=False)


def get_dataset(df):
    """
    Create example dataset from loaded dataframe.
    :param df: dataframe
    :return: dataset
    """
    return [[title, abstract] for title, abstract in zip(df.title, df.abstract)]


