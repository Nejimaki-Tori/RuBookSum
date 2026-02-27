from typing import List
from rouge import Rouge
from scipy.stats import bootstrap
from utils import AsyncList, model, extract_response
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import razdel
import os
import aiofiles
import json
import asyncio
import math
import numpy as np


nltk.download('punkt')
nltk.download('punkt_tab')

class Evaluater:
    def __init__(
        self,
        device=None, 
        encoder=None
    ):
        self.device = device
        self.encoder = encoder
        self.stemmer = SnowballStemmer('russian')       
        
    def lemmatize_text(self, text):
        tokens = word_tokenize(text, language='russian')
        return ' '.join(self.stemmer.stem(token) for token in tokens)

    def clean(self, txt):
        txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    
    def rouge_L(self, ref, pred):
        rouge = Rouge()

        ref = self.clean(ref)
        pred = self.clean(pred)

        scores = rouge.get_scores(
            hyps=[pred],  
            refs=[ref],   
            avg=False               
        )
        rouge_l = scores[0]['rouge-l']
        return rouge_l['f']

    def bertscore(self, ref, pred):
        ref = self.clean(ref)
        pred = self.clean(pred)
        ref_emb =  self.encoder.encode([s.text for s in razdel.sentenize(ref)], normalize_embeddings=True, device=self.device)
        pred_emb = self.encoder.encode([s.text for s in razdel.sentenize(pred)], normalize_embeddings=True, device=self.device)
        sims = pred_emb @ ref_emb.T
        precision = sims.max(axis=1).mean()
        recall = sims.max(axis=0).mean()
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
        
    def similarity(self, a, b):
        emb_1 = self.encoder.encode(a, device=self.device)
        emb_2 = self.encoder.encode(b, device=self.device)
    
        return round(float(self.encoder.similarity(emb_1, emb_2).item()), 3)

    def bootstrap(self, data):
        if not data:
            return None
            
        if len(data) < 2:
            return data[0], data[0], data[0]
            
        data = np.array(data)
        data1 = (data,)
        bootstrap_ci = bootstrap(
            data1, 
            np.mean, 
            confidence_level=0.95, 
            n_resamples=len(data)*100,
            random_state=42
        )
        dist = bootstrap_ci.bootstrap_distribution
        _mean = np.quantile(dist, q=0.5)
        _min = np.quantile(dist, q=0.025)
        _max = np.quantile(dist, q=0.975)
        return _mean, _min, _max

    def evaluate_annotation(self, ref_annotation, gen_annotation):
        bertscore = self.bertscore(ref_annotation, gen_annotation)
        rouge = self.rouge_L(ref_annotation, gen_annotation)
        return bertscore, rouge