#run this in /data-amoe
import pandas as pd
from sklearn.model_selection import train_test_split

path='/home/pritam.k/research/data-moe/data/tweet_eval/updated/train.csv'
df=pd.read_csv(path)
df['global_index']=df['idx']+100000
df.drop('idx', axis=1, inplace=True)
df.drop('split',axis=1,inplace=True)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.loc[train_df['difficulty'] == 1, 'label'] += 3
train_df.loc[train_df['difficulty'] == 2, 'label'] += 6
val_df.loc[val_df['difficulty'] == 1, 'label'] += 3
val_df.loc[val_df['difficulty'] == 2, 'label'] += 6
train_df.to_csv('src/MOE-DIFF_data/tweet_eval_train_diff_9labels.csv',index=False)
val_df.to_csv('src/MOE-DIFF_data/tweet_eval_val_diff_9labels.csv',index=False)


path1='/home/pritam.k/research/data-moe/data/tweet_eval/difficulty_test_instances/df_easy_test.csv'
path2='/home/pritam.k/research/data-moe/data/tweet_eval/difficulty_test_instances/df_ambi_test.csv'
path3='/home/pritam.k/research/data-moe/data/tweet_eval/difficulty_test_instances/df_hard_test.csv'

df1=pd.read_csv(path1)
df1['global_index']=df1['idx']+100000
df1.drop('idx', axis=1, inplace=True)
df1.drop('split',axis=1,inplace=True)
df1['difficulty']=0

df2=pd.read_csv(path2)
df2['global_index']=df2['idx']+200000
df2.drop('idx', axis=1, inplace=True)
df2.drop('split',axis=1,inplace=True)
df2['difficulty']=1

df3=pd.read_csv(path3)
df3['global_index']=df3['idx']+300000
df3.drop('idx', axis=1, inplace=True)
df3.drop('split',axis=1,inplace=True)
df3['difficulty']=2

combined_df = pd.concat([df1, df2, df3], ignore_index=True)
shuffled_df = combined_df.sample(frac=1)
shuffled_df.loc[shuffled_df['difficulty'] == 1, 'label'] += 3
shuffled_df.loc[shuffled_df['difficulty'] == 2, 'label'] += 6

shuffled_df.to_csv('src/MOE-DIFF_data/tweet_eval_test_diff_9labels.csv',index=False)