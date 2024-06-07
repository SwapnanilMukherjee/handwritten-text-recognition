from transformers import pipeline
import pickle
import pandas as pd 
import jiwer 
from datasets import load_metric
import spacy

cvl_data_path = '../handwritten-text-recognition/cvl-database-1-1/'
cvl_cropped_images =  cvl_data_path+'cvl-database-cropped-1-1/'
root_dir = '../handwritten-text-recognition/'

trainset_line = pd.read_csv(root_dir+'lines_trainset.csv')
testset_line = pd.read_csv(root_dir+'lines_testset.csv')

trainset_page = pd.read_csv(root_dir+'page_trainset.csv')
testset_page = pd.read_csv(root_dir+'page_testset.csv')
# some XML files are faulty and do not contain label information. XML parses returns NaN. These NaN values are replaced by the following text
testset_page.fillna('Text not available', inplace=True)



  ########################################### Spelling Correction with Transformer-based models ###############################################

fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

with open(root_dir+"recognized_text.pickle", 'rb') as f:
    output_str = pickle.load(f)

corrected_outputs = fix_spelling(output_str, max_length=20)
corrected_outputs = [x['generated_text'] for x in corrected_outputs]
print(corrected_outputs)
    
cer_metric = load_metric("cer")
corrected_cer = cer_metric.compute(predictions=corrected_outputs, references=trainset_line['text'].tolist())
corrected_wer = jiwer.wer(trainset_line['text'].tolist(), corrected_outputs)

print("CER after spelling correction:", corrected_cer)
print("WER after spelling correction:", corrected_wer)

# saving the corrected text to a pickle file 
with open(root_dir+"corrected_text.pickle", 'wb') as file:
    pickle.dump(corrected_outputs, file, protocol=pickle.HIGHEST_PROTOCOL)


'''
The CER and WER increases after spelling correction. 
This could be due to the fact that the spelling correction model is trained in english and there are many samples in
the test set which are in German. This needs to be ascertained.  The fact that CER doesn't increase a lot more is because the spell correction model replaces a word with another actual word
The CER remains high but the WER goes down
'''

  ########################################### NER with spaCy models ###############################################

# Load pre-trained SpaCy model
nlp = spacy.load("en_core_web_md") # because some texts are in German, a German language model may be more appropriate for those samples 

'''
The en_core_web_md model in spaCy recognizes the same set of entity types as the en_core_web_trf model, derived from the OntoNotes 5 dataset. 
Here are the entity types that en_core_web_md recognizes:

PERSON: People, including fictional.
NORP: Nationalities or religious or political groups.
FAC: Buildings, airports, highways, bridges, etc.
ORG: Companies, agencies, institutions, etc.
GPE: Countries, cities, states.
LOC: Non-GPE locations, mountain ranges, bodies of water.best_outputs
PRODUCT: Objects, vehicles, foods, etc. (not services).
EVENT: Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART: Titles of books, songs, paintings, etc.
LAW: Named documents made into laws.
LANGUAGE: Any named language.
DATE: Absolute or relative dates or periods.
TIME: Times smaller than a day.
PERCENT: Percentage, including “%”.
MONEY: Monetary values, including unit.
QUANTITY: Measurements, as of weight or distance.
ORDINAL: “First”, “second”, etc.
CARDINAL: Numerals that do not fall under another type.
'''

# Define a mapping from fine-grained classification to broader categories
entity_mapping = {
    "PERSON": "PERSON",
    "NORP": "ORGANIZATION",
    "FAC": "LOCATION",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "QUANTITY",
    "EVENT": "DATE",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "DATE",
    "TIME": "TIME",
    "PERCENT": "QUANTITY",
    "MONEY": "MONEY",
    "QUANTITY": "QUANTITY",
    "ORDINAL": "QUANTITY",
    "CARDINAL": "QUANTITY"
}

docs = list(nlp.pipe(corrected_outputs))

# Function to extract entities from SpaCy docs; if reduced=True, a multiple fine entities are grouped together in a larger group as defined above 
def extract_entities(docs, reduced=False):
    results = []
    for doc in docs:
        entities = []
        for ent in doc.ents:
            if reduced and ent.label_ in entity_mapping:
                entities.append({
                    "text": ent.text,
                    "label": entity_mapping[ent.label_] 
                    })
            else:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
        results.append(entities)
    return results


# Extract entities from the batch of documents
entities_list = extract_entities(docs, reduced=False)

# Print the results
for text, entities in zip(corrected_outputs, entities_list):
    print(f"Text: {text}")
    for entity in entities:
        print(f" - Entity: {entity['text']} | Type: {entity['label']}")
    print()

# saving the recognized entities to a file 
with open(root_dir+"entities_list.pickle", 'wb') as f:
    pickle.dump(entities_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                                           
print("========================================================================================================================================")                                      
                                           
                ########################################### NER with Transformer-based models ###############################################

'''
Categories recognized this model:

PER: Person
LOC: Location
ORG: Organization, company, etc.
MISC: Anything outside of above three categories
'''

# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline

# tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
# model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

# nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
# example = corrected_outputs

# ner_results = nlp(example)
# for text, entities in zip(corrected_outputs, ner_results):
#     print(f"Text: {text}")
#     for entity in entities:
#         print(f" - Entity: {entity['word']} | Type: {entity['entity_group']}")
#     print()