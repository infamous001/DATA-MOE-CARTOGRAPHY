from datasets import load_dataset
dataset = load_dataset('mteb/amazon_reviews_multi','en')

dataset['train'].to_csv('amazon/train.csv', index=False)
dataset['validation'].to_csv('amazon/val.csv', index=False)
dataset['test'].to_csv('amazon/test.csv', index=False)
from datasets import load_dataset
# df1 = load_dataset('takala/financial_phrasebank','sentences_allagree')
# df2 = load_dataset('takala/financial_phrasebank','sentences_75agree')
# df3 = load_dataset('takala/financial_phrasebank','sentences_66agree')
# df4 = load_dataset('takala/financial_phrasebank','sentences_50agree')
# from datasets import concatenate_datasets
# combined_dataset = concatenate_datasets([df1['train'], df2['train'],df3['train'],df4['train']])
# combined_dataset.to_csv('fp/train.csv', index=False)