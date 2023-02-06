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
df3 = df3.dropna(subset=['Embarked'])
# mean = df3['Age'].mean()
# df3 = df3['Age'].fillna(mean)
## (中位數)
mid = df1['Age'].median()
df1 = df1.fillna(mid)
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

## RandomForest, Logit, SCV linear,SCV RBF, K Neighbors, Gaussian NB, Decision Tree模型

#查看內容
print(df3.dtypes)
from sklearn.preprocessing import LabelEncoder
#修改格式為value值
#iloc
label=LabelEncoder()
df3.iloc[: ,4] = label.fit_transform(df3.iloc[:,4].values)
sex = df3.groupby('Sex')['Survived'].sum()
print(sex)
df3.iloc[: ,11] = label.fit_transform(df3.iloc[:,11].values)
emb = df3.groupby('Embarked')['Survived'].sum()
print(emb)
# 分成8:2
from sklearn.model_selection import train_test_split
# 被解釋變數: Survived
# 解釋變數: Pclass, Sex, Age, Embarked(刪除不用的變數)
X = df3.drop(labels=['PassengerID','Survived','Name','SibSp','Parch','Ticket','Fare','Cabin'],axis=1)
y = df3['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# 標準化(讓資料分布限縮在-1~1不那麼分散,特徵縮放通常可以選擇不使用)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# 訓練隨機森林解決回歸問題
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=25, random_state=0)
rfc.fit(X_train, y_train) #開始訓練
y_pre = rfc.predict(X_test) #預測結果
print('測試集分數:',rfc.score(X_test,y_test))
print('訓練集分數:',rfc.score(X_train,y_train))
print('特徵重要程度:',rfc.feature_importances_)
# 評估回歸性能
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pre))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pre))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test,y_pre)))