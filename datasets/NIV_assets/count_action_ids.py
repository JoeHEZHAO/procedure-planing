import os
import csv
from csv import reader
from collections import Counter
import pickle

files = os.listdir("./csvs")
csv_files = [x for x in files if x.endswith(".csv")]

all_action_name = []

for csv in csv_files:
    # open file in read mode
    with open(os.path.join("./csvs", csv), "r") as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            all_action_name.append(csv.split("_")[0] + row[0])

action_counter = Counter(all_action_name)
print(action_counter)
print(len(action_counter))

action_dict = {}
"Loop through counter"
for i, k in enumerate(action_counter.keys()):
    action_dict[k] = i
print(action_dict)

with open("full_action_ids.pickle", "+wb") as f:
    pickle.dump(action_dict, f)
