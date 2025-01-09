# import pandas as pd
# from sklearn.model_selection import train_test_split
# # df1=pd.read_csv('/home/pritam.k/research/data-moe/data/sst/train.csv')
# # df1['global_idx']=df1['idx']+100000
# # df2=pd.read_csv('/home/pritam.k/research/data-moe/data/sst/val.csv')
# # df2['global_idx']=df2['idx']+100000
# # df3=pd.read_csv('/home/pritam.k/research/data-moe/data/sst/test.csv')
# # df3['global_idx']=df3['idx']+100000

# # df1.to_csv('/home/pritam.k/research/data-moe/data/sst/train.csv',index=False)
# # df2.to_csv('/home/pritam.k/research/data-moe/data/sst/val.csv',index=False)
# # df3.to_csv('/home/pritam.k/research/data-moe/data/sst/test.csv',index=False)
# train_df=pd.read_csv('/home/pritam.k/research/data-moe/data/amazon/train.csv')
# val_df=pd.read_csv('/home/pritam.k/research/data-moe/data/amazon/val.csv')
# test_df=pd.read_csv('/home/pritam.k/research/data-moe/data/amazon/test.csv')
# # # val_df=pd.read_csv('/home/pritam.k/research/data-moe/data/snli/val.csv')
# # test_df=pd.read_csv('/home/pritam.k/research/data-moe/data/snli/test.csv')
# # train_val, test_df= train_test_split(df,test_size=0.1, random_state=42)
# # train_df,val_df=train_test_split(train_val,test_size=0.1, random_state=42)

# train_df['idx']=train_df.index
# train_df['global_idx']=train_df['idx']+1000000
# val_df['idx']=val_df.index
# val_df['global_idx']=val_df['idx']+1000000
# test_df['idx']=test_df.index
# test_df['global_idx']=test_df['idx']+1000000
# # train_df['premise']=pd.Series([str(p) for p in train_df['premise']])
# # train_df['hypothesis']=pd.Series([str(p) for p in train_df['hypothesis']])

# # val_df['premise']=pd.Series([str(p) for p in val_df['premise']])
# # val_df['hypothesis']=pd.Series([str(p) for p in val_df['hypothesis']])

# # test_df['premise']=pd.Series([str(p) for p in test_df['premise']])
# # test_df['hypothesis']=pd.Series([str(p) for p in test_df['hypothesis']])


# train_df.to_csv('/home/pritam.k/research/data-moe/data/amazon/train.csv',index=False)
# val_df.to_csv('/home/pritam.k/research/data-moe/data/amazon/val.csv',index=False)
# test_df.to_csv('/home/pritam.k/research/data-moe/data/amazon/test.csv',index=False)

# import pandas as pd
# df1=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/difficulty_test_instances/df_ambi_test.csv")
# df1['diff']=1
# df2=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/difficulty_test_instances/df_easy_test.csv")
# df2['diff']=0
# df3=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/difficulty_test_instances/df_hard_test.csv")
# df3['diff']=2
# df_concatenated = pd.concat([df1, df2, df3], ignore_index=True)
# df = df_concatenated.sample(frac=1, random_state=42).reset_index(drop=True)

# df.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/merged_diff_test.csv",index=False)
# import pandas as pd
# from sklearn.model_selection import train_test_split

# df1=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/merged_diff_test.csv")
# train_df, temp_df = train_test_split(df1, test_size=0.2, random_state=42)
# val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
# train_df.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_train.csv")
# val_df.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_val.csv")
# test_df.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_test.csv")
import pandas as pd

df1=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_train.csv")
# df1.loc[df1['diff'] == 0, 'label'] += 0
# df1.loc[df1['diff'] == 1, 'label'] += 3
# df1.loc[df1['diff'] == 2, 'label'] += 6
df1["global_idx"] = range(10000, 10000 + len(df1))

df2=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_val.csv")
# df2.loc[df2['diff'] == 0, 'label'] += 0
# df2.loc[df2['diff'] == 1, 'label'] += 3
# df2.loc[df2['diff'] == 2, 'label'] += 6
df2["global_idx"] = range(20000, 20000 + len(df2))

df3=pd.read_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_test.csv")
# df3.loc[df3['diff'] == 0, 'label'] += 0
# df3.loc[df3['diff'] == 1, 'label'] += 3
# df3.loc[df3['diff'] == 2, 'label'] += 6
df3["global_idx"] = range(30000, 30000 + len(df3))


df1.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_train.csv",index=False)
df2.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_val.csv",index=False)
df3.to_csv("/home/pritam.k/research/data-moe/data/tweet_eval/m_test.csv",index=False)
