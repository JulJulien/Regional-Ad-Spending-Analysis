import seaborn as sns
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature_distributions(df, cols=None, show_shapiro=True):
    """
    Plots bell curve (histogram + KDE) for numeric columns to check normality.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe (wide format).
    cols : list, optional
        List of column names to plot. Defaults to all numeric columns.
    show_shapiro : bool, optional
        Whether to print Shapiro-Wilk test results (default=True).

    Returns
    -------
    pd.DataFrame
        Shapiro-Wilk test results for each column if show_shapiro=True.
    """

    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    results = []

    for col in cols:
        data = df[col].dropna()
        if data.empty:
            continue

        # ---- Plot distribution ----
        plt.figure(figsize=(6, 4))
        sns.histplot(data, kde=True, bins=30, color='steelblue', edgecolor='black')
        plt.title(f"Distribution of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ---- Shapiro-Wilk test ----
        if show_shapiro:
            stat, p = shapiro(data)
            results.append({'Feature': col, 'W-Statistic': stat, 'p-value': p})

    if show_shapiro:
        results_df = pd.DataFrame(results)
        print("\n--- Shapiro-Wilk Test Results ---")
        print(results_df.sort_values('p-value', ascending=False))

        return results_df
