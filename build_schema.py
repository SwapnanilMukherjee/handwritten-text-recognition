import json
import pickle 

cvl_data_path = '../handwritten-text-recognition/cvl-database-1-1/'
cvl_cropped_images =  cvl_data_path+'cvl-database-cropped-1-1/'
root_dir = '../handwritten-text-recognition/'

'''
metadata can be easily created out of this by defining some entities as meta entities such as DATE, TIME, LOCATION, ORGANIZATION. They will be recorded as separate fields,
and all other types entities will be clubbed under 'entities'. But special care needs to be taken such that multiple of these entities if detected by the model do not all get stored 
as metadata.
'''

with open(root_dir+'entities_list.pickle', 'rb') as file:
    entities_list = pickle.load(file)

with open(root_dir+'corrected_text.pickle', 'rb') as file:
    corrected_outputs = pickle.load(file)
    
data_schema = []

for text, entities in zip(corrected_outputs, entities_list):
    sample = {
        "text": text,
        "entities": {}
    }
    for entity in entities:
        if entity['label'] not in list(sample['entities'].keys()):
            sample['entities'][entity['label']] = []
        sample['entities'][entity['label']].append(entity['text'])
    data_schema.append(sample)

json_schema = json.dumps(data_schema, indent=4)
print(json_schema)


# saving the combined schema for all the samples as a JSON file 
with open(root_dir+"schema.json", 'w') as file:
    json.dump(data_schema, file, indent=4)