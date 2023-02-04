import pandas as pd
import numpy
#讀入.csv
df1 = pd.read_csv('C:/Big Data/Competitions/Titanic/test.csv')
df2 = pd.read_csv('C:/Big Data/Competitions/Titanic/gender_submission.csv')
#多表連結 用test.csv為基底
# concat->將兩DataFrame直接合併
# pd.concat([df1,df2], join='outer', ignore_index=True).to_csv('C:/Big Data/Competitions/Titanic/concat_out.csv')
# merge->依照某列合併兩個DataFrame
inner_join=pd.merge(left=df1,right=df2,how='inner').to_csv('C:/Big Data/Competitions/Titanic/inner.csv')