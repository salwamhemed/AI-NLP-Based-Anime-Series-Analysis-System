import spacy
from nltk import sent_tokenize
import os
import pandas as pd
from utils import load_subtitles
from ast import literal_eval
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))


class NamedEntityRecognizer:
    def __init__(self):
        self.model = self.load_model()
        pass

    def load_model(self):
     nlp = spacy.load("en_core_web_trf")
     return nlp
    
    def get_ners_inference(self,script):
     script_sentences = sent_tokenize(script)
 
     ner_output = []

     for sentence in script_sentences:
        doc = self.load_model(sentence)
        ners = set()
        for entity in doc.ents:
            if entity.label_ =="PERSON":
                full_name = entity.text
                first_name = entity.text.split(" ")[0]
                first_name = first_name.strip()
                ners.add(first_name)
        ner_output.append(ners)

     return ner_output
     


    def get_ners(self,dataset_path,save_path=None):
        
     if save_path is not None and os.path.exists(dataset_path) :
        df = pd.read_csv(dataset_path)
        df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
        return df
     
     #Get Script
     df = load_subtitles(dataset_path)
     #Run Inference
     df['ners']= df['scripts'].apply(self.get_ners_inference)
     if save_path is not None:
       df.to_csv(save_path,index=False)
        
     return df

