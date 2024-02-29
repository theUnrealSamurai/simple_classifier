from glob import glob 
import pandas as pd


df1 = pd.read_csv("/Users/mits-mac-001/Code/simple_classifier/sales_forecasting/data/株式会社開花者_20220401-20220930.csv", encoding='unicode_escape')
df2 = pd.read_csv("/Users/mits-mac-001/Code/simple_classifier/sales_forecasting/data/株式会社開花者_20221001-20230531.csv", encoding='unicode_escape')

print(df1.head())
print(df2.head())


