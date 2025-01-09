import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


csv_file = '/home/pritam.k/research/data-moe/data/imdb/train_imdb.csv' 
df = pd.read_csv(csv_file)


tokenizer = AutoTokenizer.from_pretrained('roberta-base')

max_seq_len = 512  # Maximum sequence length

df['token_count'] = df['text'].apply(lambda x: len(tokenizer.encode(x, max_length=max_seq_len, truncation=True)))

df.to_csv("checking.csv")

plt.figure(figsize=(10, 6))
plt.hist(df['token_count'], bins=30, color='blue', alpha=0.7,edgecolor='white')
plt.title('Distribution of Token Counts')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.savefig("/home/pritam.k/research/data-moe/src/plots/imdb_train_token_distribution.png")
plt.show()
