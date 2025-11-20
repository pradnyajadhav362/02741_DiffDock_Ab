# sample 40 random structures from full aadam dataset for quick demo training
# creates new directory with splits and csv file

import random
import os

project_root = "/path/to/project"
full_csv = f"{project_root}/datasets/AADaM_full/_all_ids.csv"
demo_dir = f"{project_root}/datasets/demo40"

random.seed(42)

print("reading full dataset")
with open(full_csv, 'r') as f:
    lines = f.readlines()

header = lines[0]
data_lines = lines[1:]

print(f"total structures: {len(data_lines)}")

sampled = random.sample(data_lines, 40)

train_ids = [line.split(',')[0] for line in sampled[:32]]
val_ids = [line.split(',')[0] for line in sampled[32:]]

os.makedirs(f"{demo_dir}/splits", exist_ok=True)

with open(f"{demo_dir}/_all_ids.csv", 'w') as f:
    f.write(header)
    for i, line in enumerate(sampled):
        parts = line.strip().split(',')
        split = 'train' if i < 32 else 'val'
        f.write(f"{parts[0]},{split}\n")

with open(f"{demo_dir}/splits/demo_train.txt", 'w') as f:
    f.write('\n'.join(train_ids) + '\n')

with open(f"{demo_dir}/splits/demo_val.txt", 'w') as f:
    f.write('\n'.join(val_ids) + '\n')

print(f"created demo dataset with 32 train + 8 val structures")
print(f"saved to {demo_dir}")


