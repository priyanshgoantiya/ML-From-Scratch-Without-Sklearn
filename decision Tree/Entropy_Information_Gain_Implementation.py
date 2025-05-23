import numpy as np
def calculate_entropy(data):
  total_count=len(data)
  label_counts={}
  for row in data:
    label=row[-1]
    if label not in label_counts:
      label_counts[label]=0
    label_counts[label]+=1
  entropy=0
  for count in label_counts.values():
    probability=count/total_count
    entropy-=probability*np.log2(probability)
  return entropy 
def Information_gain(data,feature_index):
  entropy_before_split=calculate_entropy(data)
  unique_features=set(row[feature_index] for row in data)
  entropy_after_split = 0
  for feature in unique_features:
    subset=[row for row in data if row[feature_index]==feature]
    subset_entropy=calculate_entropy(subset)
    weight=len(subset)/len(data)
    entropy_after_split+=(weight*subset_entropy)
  return entropy_before_split-entropy_after_split
data = [
    ['Sunny', 85, 85, False, 'High', 'Weak', 'No', 5.5, 'Mild', 'Dry', 'No'],
    ['Sunny', 80, 90, True, 'High', 'Weak', 'Yes', 6.0, 'Hot', 'Humid', 'No'],
    ['Overcast', 83, 86, False, 'High', 'Strong', 'No', 5.0, 'Mild', 'Dry', 'Yes'],
    ['Rainy', 70, 96, False, 'Normal', 'Strong', 'Yes', 4.5, 'Cool', 'Wet', 'Yes'],
    ['Rainy', 68, 80, False, 'Normal', 'Weak', 'No', 5.8, 'Cool', 'Humid', 'Yes'],
    ['Rainy', 65, 70, True, 'Normal', 'Weak', 'Yes', 6.2, 'Cool', 'Humid', 'No'],
    ['Overcast', 64, 65, True, 'High', 'Strong', 'No', 6.4, 'Mild', 'Dry', 'Yes'],
    ['Sunny', 72, 95, False, 'High', 'Weak', 'No', 5.1, 'Hot', 'Humid', 'No'],
    ['Sunny', 69, 70, False, 'Normal', 'Strong', 'Yes', 4.9, 'Cool', 'Dry', 'Yes'],
    ['Rainy', 75, 80, False, 'Normal', 'Strong', 'No', 6.3, 'Mild', 'Wet', 'Yes']
]
IG=Information_gain(data,3)
print(IG)
