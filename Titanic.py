import sys
sys.path.append('C:/Big Data/Competitions/Titanic/venv/Lib/site-packages')
import pandas as pd
import matplotlib.pyplot as plt
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

##Embarked倖存總數
emb = df3.groupby('Embarked')['Survived'].sum()
print('倖存人數:',emb)
##計算Embarked倖存率
emb = df3.groupby('Embarked')['Survived'].mean()
print('各登入口倖存率:',emb)

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

##計算年齡10歲為一區間包含10歲
age_cut = pd.cut(df3['Age'],bins=range(0,100,10),right=True)
age = df3.groupby(age_cut)['Survived'].sum()
print(age)
age = df3.groupby(age_cut)['Survived'].mean()
print(age)
##計算年齡自訂區間
age_cut = pd.cut(df3['Age'],[0,10,20,30,40,50,60],right=True)
age = df3.groupby(age_cut)['Survived'].sum()
print(age)
age = df3.groupby(age_cut)['Survived'].mean()
print(age)

##Pclass
Pc = df3.groupby('Pclass')['Survived'].sum()
print(Pc)
Pc = df3.groupby('Pclass')['Survived'].mean()
print(Pc)

##Sex
sex = df3.groupby('Sex')['Survived'].sum()
print(sex)
sex = df3.groupby('Sex')['Survived'].mean()
print(sex)

##Sex, Pclass, Embarked, Age
TT=df3.pivot_table('Survived',['Sex',age_cut],'Pclass')
print(TT)
TT=df3.pivot_table('Survived',['Sex',age_cut],'Embarked')
print(TT)
#折線圖
# Survived_ratio = [s / len(df3['Survived']) for s in (df3['Survived'])]
# plt.plot(df3['Age'], Survived_ratio)
#直方圖
# plt.hist(df3['Survived'], bins=5)
#x,y軸及標題
# plt.xlabel('Age')
# plt.ylabel('Survived Ratio')
# plt.title('Survived Ratio of Titanic by Age')
# plt.show()

