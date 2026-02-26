from typing import List
from rouge import Rouge
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


nltk.download('punkt')
nltk.download('punkt_tab')

QUESTIONS_COVERAGE_PROMPT = """Ты - эксперт в оценивании качества аннотаций для книг. Твоя задача — тщательно оценить, насколько представленная аннотация позволяет ответить на конкретный вопрос, касающийся ключевых аспектов исходного произведения.

Вопрос:
{question}
Текст аннотации:
{text}

Содержится ли в этом тексте ответ на вопрос?
Начни ответ с {yes}, если содержится или с {no}, если не содержится.
"""

GOLD_QUESTIONS_PROMPT = """На основе данной аннотации сформируй несколько ключевых вопросов, ответы на которые можно однозначно дать, зная содержание аннотации. Пиши каждый вопрос с новой строки, пронумеровать не нужно. Ничего кроме вопросов — ни вступлений, ни пояснений, ни заголовков. 

Аннотация:
---
{ref_annotation}
---
Ключевые вопросы нужно писать по порядку, начиная каждый вопрос с новой строки. Кроме ключевых вопросов ничего писать не нужно.
"""

ANSWER_PROMPT = """На основе данной аннотации сформируй ответ на поставленный вопрос. Твоя задача — **строго на основе предоставленной аннотации** сформировать ответ на ключевой вопрос.

Аннотация:
{ref_annotation}

Ключевой вопрос:
{key_q}
---

Не пиши вопрос, пиши только ответ.
"""

class Evaluater:
    def __init__(
        self, 
        #evaluater=None, 
        device=None, 
        encoder=None
    ):
        self.device = device
        self.encoder = encoder
        self.stemmer = SnowballStemmer('russian')
        #self.client_eval = evaluater
        #self.qa_dir = 'qa_data'        
        
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
        rouge_l = scores[0]["rouge-l"]
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

    def evaluate_annotation(self, ref_annotation, gen_annotation):
        bertscore = self.bertscore(ref_annotation, gen_annotation)
        rouge = self.rouge_L(ref_annotation, gen_annotation)
        return bertscore, rouge