import json
import numpy as np

file_path = "left.json"

# Open and load the JSON file
with open(file_path, "r") as file:
    dataleft = json.load(file)

data = dataleft[0]


np.savez('pointsForReconstruction.npz', pts1=np.array(list(data.values()))[:,:-1])