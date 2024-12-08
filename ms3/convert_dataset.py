import csv
import json
import os

def convert_csv_to_json(csv_path):
    # Dictionary to map age_groups and gender to indices
    age_group_map = {}
    gender_map = {"m": 0, "f": 1}
    
    labels = []
    
    with open(csv_path, 'r') as csvfile:
        # Skip header
        next(csvfile)
        reader = csv.reader(csvfile, delimiter=';')
        
        for row in reader:
            image_path, age_group, gender = row
            
            # changing last 3 chars from jpg to png
            image_path = image_path[:-3] + 'png'
            
            # Create index for age_group if not exists
            if age_group not in age_group_map:
                age_group_map[age_group] = len(age_group_map)
            
            # Create label array [age_group_index, gender_index]
            label = [age_group_map[age_group], gender_map[gender.lower()]]
            
            # Add to labels list
            labels.append([image_path, label])
    
    # Create final JSON structure
    dataset = {"labels": labels}
    
    # Write to JSON file
    with open('dataset.json', 'w') as jsonfile:
        json.dump(dataset, jsonfile, indent=4)
    
    print(f"Age group mapping: {age_group_map}")
    return dataset

if __name__ == "__main__":
    csv_path = '../train/faces_anno_train.csv'
    convert_csv_to_json(csv_path)
