import pandas as pd
#讀入.csv
df1 = pd.read_excel('C:/Big Data/Competitions/Titanic/test.csv')
df2 = pd.read_excel('C:/Big Data/Competitions/Titanic/gender_submission.csv')
#多表連結 用test.csv為基底
pd.concat([df1,df2], join='left', ignore_index=True).to_csv('C:/Big Data/Competitions/Titanic/left.csv')