import torch
from transformers import pipeline
import numpy as np
from nltk import sent_tokenize
import pandas as pd
import os
import sys
import pathlib
from utils import load_subtitles

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))

class theme_classifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    def load_model(self, device):
        theme_classifier = pipeline("zero-shot-classification", model=self.model_name, device=device)
        return theme_classifier

    def get_themes(self, script):
        script_sentences = sent_tokenize(script)

        # Batch sentences into groups of 20 sentences
        sentences_batch_size = 20
        script_batches = []
        for i in range(0, len(script_sentences), sentences_batch_size):
            sent = " ".join(script_sentences[i:i + sentences_batch_size])
            script_batches.append(sent)

        # Run the model on all batches
        theme_output = self.theme_classifier(script_batches, self.theme_list, multi_label=True)

        # Modify the output format
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        # Calculating the mean value for each theme
        themes = {key: np.mean(np.array(value)) for key, value in themes.items()}
        return themes

    def get_themes_result(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df

        # Load the dataset (ensure your `load_subtitles` function works)
        df = load_subtitles(dataset_path)
        
        # Apply the classifier
        output_themes = df['scripts'].apply(self.get_themes)
        themes_df = pd.DataFrame(output_themes.tolist())
        
        # Combine the theme scores with the original DataFrame
        df = pd.concat([df, themes_df], axis=1)

        # Save output
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df

