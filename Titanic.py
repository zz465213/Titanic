import pandas as pd
import numpy as np
#讀入.csv
df1 = pd.read_csv('C:/Big Data/Competitions/Titanic/test.csv')
df2 = pd.read_csv('C:/Big Data/Competitions/Titanic/gender_submission.csv')
df3 = pd.read_csv('C:/Big Data/Competitions/Titanic/train.csv')

## merge->依照某列合併兩個DataFrame
df4 = pd.merge(left=df1,right=df2,how='inner')
df4.to_csv('C:/Big Data/Competitions/Titanic/check.csv')
## concat->將兩DataFrame直接合併,不保留原本index
# df3_4 = pd.concat([df3,df4],join='outer', ignore_index=True)
## 根據值排序
# df3_4 = df3_4.sort_values(by=["PassengerID"],ascending=True)

##查找空值
# na = df1['Age'].isna()
# print(na)
## 處理空值(均值)
# mean = df1['Age'].mean()
# df1 = df1['Age'].fillna(mean)
## (中位數)
mid = df1['Age'].median()
mid1 = df1['Age'].fillna(mid)
# mid1.to_csv('C:/Big Data/Competitions/Titanic/test1.csv')
mid = df3['Age'].median()
mid3 = df3['Age'].fillna(mid)
# mid3.to_csv('C:/Big Data/Competitions/Titanic/train1.csv')

##計算Embarked
# Q
Q_total = (df3['Embarked'] == 'Q').sum()
print('Q登入總人數:',Q_total)
Q_Survived = ((df3['Survived'] == 1) & (df3['Embarked'] == 'Q')).sum()
print('Q登入倖存人數:',Q_Survived)
Q_ratio = round(float(Q_Survived / Q_total),2)
print('Q登入倖存比:',Q_ratio)
# S
S_total = (df3['Embarked'] == 'S').sum()
print('S登入總人數',S_total)
S_Survived = ((df3['Survived'] == 1) & (df3['Embarked'] == 'S')).sum()
print('S登入倖存人數',S_Survived)
S_ratio = round(float(S_Survived / S_total),2)
print('S登入倖存比:',S_ratio)
# C
C_total = (df3['Embarked'] == 'C').sum()
print('C登入總人數',C_total)
C_Survived = ((df3['Survived'] == 1) & (df3['Embarked'] == 'C')).sum()
print('C登入倖存人數',C_Survived)
C_ratio = round(float(C_Survived / C_total),2)
print('C登入倖存比:',C_ratio)

##年齡

##Pclass
