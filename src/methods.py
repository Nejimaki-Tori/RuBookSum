import json
from blueprint import Blueprint
from hierarchical import Hierarchical
from iterative import Iterative
from pseudo import Pseudo
from utils import LlmCompleter, chunk_text
from metrics import Evaluater
import time
import pandas as pd


class Summarisation:
    def __init__(
        self, 
        key: str = '', 
        url: str = '', 
        device=None, 
        encoder=None, 
        model_name: str = '', 
        is_thinking_needed: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.encoder = encoder
        self.think_pass = '' if is_thinking_needed else ' /no_think'
        self.client = LlmCompleter(url, key, self.model_name)
        self.blueprint = Blueprint(
            self.client, 
            self.device, 
            self.encoder, 
            mode='default', 
            think_pass=self.think_pass
        )
        self.hierarchical = Hierarchical(
            self.client, 
            self.device, 
            self.encoder, 
            mode='default',
            think_pass=self.think_pass
        )
        #self.client_evaluater = LlmCompleter(URL, KEY, 'Qwen3-235B-A22B-Instruct-2507')
        self.evaluater = Evaluater(
            #evaluater=self.client_evaluater, 
            device=self.device, 
            encoder=self.encoder
        )
        
    def change_model(self, key: str = '', url: str = '', model_name: str = '', is_thinking_needed: bool = False):
        self.model_name = model_name
        self.think_pass = '' if is_thinking_needed else ' /no_think'
        self.client = LlmCompleter(url, key, self.model_name)
        self.blueprint = Blueprint(self.client, self.device, self.encoder, mode='default', think_pass=self.think_pass)
        self.hierarchical = Hierarchical(self.client, self.device, self.encoder, think_pass=self.think_pass)

    def prepare_enviroment(self):
        try:
            with open('combined_data.json', 'r', encoding='utf-8') as f:
                self.collection = json.load(f)
        except:
            raise ValueError('No file with annotations and book texts!')

    async def run_method(self, text: str, method: str = '', mode: str = '', initial_word_limit: int = 500):
        if method not in ('blueprint', 'hierarchical'):
            raise ValueError(f'No such method: {method}! Only blueprint and hierarchical are available.')

        if method == 'blueprint':
            if mode not in ('default', 'cluster'):
                raise ValueError(f'No such mode: {mode}! Only default and cluster are available for blueprint.')

            chunks = chunk_text(text)
            summary = await self.blueprint.run(chunks=chunks, initial_word_limit=initial_word_limit, mode=mode)
            return summary

        if method == 'hierarchical':
            if mode not in ('default', 'filtered'):
                raise ValueError(f'No such mode: {mode}! Only default and filtered are available for hierarchical.')

            chunks = chunk_text(text)
            summary = await self.hierarchical.run(chunks, initial_word_limit=500, mode=mode)
            return summary

    def evaluate_annotation(self, ref_annotation, gen_annotation):
        return self.evaluater.evaluate_annotation(ref_annotation, gen_annotation)

    def append_to_json(self, record: dict, json_path: str = 'benchmark_results.json'):
        data = []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = []

        data.append(record)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def run_benchmark_one_method(
        self, 
        is_evalutation_needed: bool = False,
        number_of_books: int = 3, 
        method: str = 'hierarchical', 
        mode: str = 'default', 
        initial_word_limit: int = 500,
        text_length_cap: int = 80000,
        save_json_path: str = 'benchmark_results.json'
    ):
        rows = []
        for idx, item in enumerate(self.collection[:number_of_books]):
            if text_length_cap != -1 and len(item['text']) > text_length_cap:
                continue

            start_timer = time.perf_counter()
            summary = await self.run_method(text=item['text'], method=method, mode=mode, initial_word_limit=initial_word_limit)
            end_timer = time.perf_counter()
            runtime = end_timer - start_timer

            generated_annotation = summary
            gold_annotation = item['annotation']
            
            if is_evalutation_needed:
                bertscore, rouge = self.evaluate_annotation(generated_annotation, gold_annotation) 
    
                record = {
                    'book_idx': idx,
                    'book_title': item['title'],
                    'method': method,
                    'mode': mode,
                    'initial_word_limit': initial_word_limit,
                    'text_len': len(item['text']),
                    'runtime_sec': round(runtime, 4),
                    'bertscore_p': float(bertscore[0]),
                    'bertscore_r': float(bertscore[1]),
                    'bertscore_f': float(bertscore[2]),
                    'rougeL': float(rouge),
                }
    
                rows.append(record)
    
                self.append_to_json(record, json_path=save_json_path)
            else: 
                record = {
                    'book_idx': idx,
                    'book_title': item['title'],
                    'method': method,
                    'mode': mode,
                    'initial_word_limit': initial_word_limit,
                    'text_len': len(item['text']),
                    'runtime_sec': round(runtime, 4),
                    'generated_annotation': generated_annotation,
                    'gold_annotation': gold_annotation,
                }

                rows.append(record)
                self.append_to_json(record, json_path=save_json_path)
        
        if is_evalutation_needed:    
            data = pd.DataFrame(rows)
            print(data)

        return rows