import pandas as pd
import numpy as np


df = pd.read_csv("data_translated/株式会社開花者_20220401-20220930 - 株式会社開花者_20220401-20220930.csv")

# df = df[["Sales Time", "Ticket Type", "Sales Amount", "Item Name", "Group Name"]]
df = df[["Sales Time", "Ticket Type", "Sales Amount", "Group Name"]]


df['Sales Time'] = pd.to_datetime(df['Sales Time'])

# Extract year, month, day, and weekday features
df['year'] = df['Sales Time'].dt.year
df['month'] = df['Sales Time'].dt.month
df['day'] = df['Sales Time'].dt.day
df['weekday'] = df['Sales Time'].dt.weekday  # Monday=0, Sunday=6
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# df["Total Sales"] = df.groupby(["day", "month"])["Group Name"].count()
# df['Total Sales'] = df.groupby(['month'])['day'].transform('nunique')
x = df.groupby(['year', 'month', 'day']).size().reset_index(name='sales_count')
# df["Total Sales"] = x["sales_count"]


# print(df.tail(60))

print(df.columns)