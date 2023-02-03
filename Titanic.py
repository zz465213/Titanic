import pandas as pd
import numpy
#讀入.csv
df1 = pd.read_csv('C:/Big Data/Competitions/Titanic/test.csv')
df2 = pd.read_csv('C:/Big Data/Competitions/Titanic/gender_submission.csv')
#多表連結 用test.csv為基底
# concat就是將兩個資料直接合併
# pd.concat([df1,df2], join='inner', ignore_index=True).to_csv('C:/Big Data/Competitions/Titanic/inner.csv')
# merge自己找共有項
inner_join=pd.merge(left=df1,right=df2,how='inner').to_csv('C:/Big Data/Competitions/Titanic/inner.csv')