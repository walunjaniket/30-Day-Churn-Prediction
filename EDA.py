#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 07:57:17 2025

@author: aniket
"""
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataframe from the saved csv file
df = pd.read_csv("cleaned_dataframe.csv")

df["invoice_date"] = pd.to_datetime(df["invoice_date"])

# 1. Basic info
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nUnique customers:", df['customer_id'].nunique())
print("\nDate range:", df['invoice_date'].min(),
      "to", df['invoice_date'].max())

## Plots
# Distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Amount distribution
axes[0, 0].hist(df["total_amount"], bins=50, log=True)
axes[0, 0].set_title("Total Amount Distribution (log Scale)")
axes[0, 0].tick_params(axis="x", rotation=15)

# Quantity distribution
axes[0, 1].hist(df["quantity"], bins=30, log=True)
axes[0, 1].set_title("Quantity Distribution (log Scale)")
axes[0, 1].tick_params(axis="x", rotation=15)

# Purchases per customer
purchases_per_cust = df.groupby("customer_id")["invoice"].nunique()
axes[1, 0].hist(purchases_per_cust, bins=30, log=True)
axes[1, 0].set_title("Purchases Per Customer (log Scale)")

# Country count
country_count = df["country"].value_counts()
axes[1, 1].bar(country_count.index, country_count.values)
axes[1, 1].set_title("Country Count")
axes[1, 1].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.savefig("EDA_plots_1.png")
plt.show()

# Correlation plot for numeric features
num_df = df[["quantity", "price", "total_amount"]]
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("feature_correlation_map.png")
plt.show()

# Time series (monthly sales)
df["month"] = df["invoice_date"].dt.to_period("M")
monthly_sales = df.groupby("month")["total_amount"].sum()
monthly_sales.plot(kind="line")
plt.title("Monthly Sales")
plt.savefig("monthly_sales.png")
plt.show()

# Outlier detection
sns.boxplot(data=num_df)
plt.title("Boxplot of Numerical Features")
plt.tight_layout()
plt.savefig("Boxplot_for_outliers")
plt.show()
