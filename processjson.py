import json
import numpy as np

file_path = "left.json"

# Open and load the JSON file
with open(file_path, "r") as file:
    dataleft = json.load(file)

file_path = "right.json"

# Open and load the JSON file
with open(file_path, "r") as file:
    dataright = json.load(file)

def store_npz (dict1, dict2):
    pts1=[]
    pts2=[]
    for key in dict1:
        if dict2[key][-1] > 0.1 and dict1[key][-1] > 0.1:
            pts1.append(dict1[key][:-1])
            pts2.append(dict2[key][:-1])
    np.savez('some_corresp.npz', pts1=pts1, pts2=pts2)

store_npz(dataleft[0], dataright[0])