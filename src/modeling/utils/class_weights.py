import pandas as pd


def compute_class_weights(df: pd.DataFrame, column: str):
    """
    Uses sklearn formula for balanced class weights.
    class_weight = total_num_samples / (n_classes * count_per_class)
    """
    class_frequency = df[column].value_counts().sort_index()
    total_num_samples = class_frequency.sum()
    n_classes = len(class_frequency)
    class_weight = total_num_samples / (n_classes * class_frequency)
    return class_weight
