import pandas as pd
import matplotlib.pyplot as plt


csv_file_path = '/home/pritam.k/research/data-moe/src/csv_files/expert_dist_CBZ.csv'
data = pd.read_csv(csv_file_path)


plt.figure(figsize=(8, 5))
plt.bar(data['expert'], data['num_of_mapped_instances'], color='blue')


plt.xlabel('expert')
plt.ylabel('num_instances_mapped')
plt.title('expert distribution only CE')
plt.xticks(data['expert'])  


plt.grid(axis='y')  
plt.savefig('plots/expert_dist_CBZ.png')
plt.show()