import pandas as pd
import json

def calculate_overlap_percentage(expert_set, diff_set):
    overlap = expert_set.intersection(diff_set)
    overlap_percentage = (len(overlap) / len(expert_set)) * 100
    return overlap_percentage

path1="/home/pritam.k/research/data-moe/results/expert_mapped_3_cbz_diff_final_tweet_eval.json"
path2="/home/pritam.k/research/data-moe/src/MOE-DIFF_data/tweet_eval_test_diff.csv"

with open(path1, 'r') as file:
    data = json.load(file)

idx_mapped_expert0=data['0']
idx_mapped_expert0=set(idx_mapped_expert0)
idx_mapped_expert1=data['1']
idx_mapped_expert1=set(idx_mapped_expert1)
idx_mapped_expert2=data['2']
idx_mapped_expert2=set(idx_mapped_expert2)

df=pd.read_csv(path2)

idx_easy=df.loc[df['difficulty'] == 0, 'global_index']
idx_easy=idx_easy.tolist()
print(len(idx_easy))
idx_easy=set(idx_easy)
print(len(idx_easy))
idx_ambi=df.loc[df['difficulty'] == 1, 'global_index']
idx_ambi=idx_ambi.tolist()
idx_easy=set(idx_ambi)
idx_hard=df.loc[df['difficulty'] == 2, 'global_index']
idx_hard=idx_hard.tolist()
idx_easy=set(idx_hard)

print("expert:0->")
print("easy")
expert0_easy=calculate_overlap_percentage(idx_easy,idx_easy)
print(expert0_easy)
print("ambi")
expert0_ambi=calculate_overlap_percentage(idx_mapped_expert0,idx_ambi)
print(expert0_ambi)
print("hard")
expert0_hard=calculate_overlap_percentage(idx_mapped_expert0,idx_hard)
print(expert0_hard)

print("expert:1->")
print("easy")
expert1_easy=calculate_overlap_percentage(idx_mapped_expert1,idx_easy)
print(expert1_easy)
print("ambi")
expert1_ambi=calculate_overlap_percentage(idx_mapped_expert1,idx_ambi)
print(expert1_ambi)
print("hard")
expert1_hard=calculate_overlap_percentage(idx_mapped_expert1,idx_hard)
print(expert1_hard)

print("expert:2->")
print("easy")
expert2_easy=calculate_overlap_percentage(idx_mapped_expert2,idx_easy)
print(expert2_easy)
print("ambi")
expert2_ambi=calculate_overlap_percentage(idx_mapped_expert2,idx_ambi)
print(expert2_ambi)
print("hard")
expert2_hard=calculate_overlap_percentage(idx_mapped_expert2,idx_hard)
print(expert2_hard)


