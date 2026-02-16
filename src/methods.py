import json
from blueprint import Blueprint
from hierarchical import Hierarchical
from utils import LlmCompleter, chunk_text
from metrics import Evaluater
import time
import pandas as pd
from pathlib import Path


METHOD_BLUEPRINT = 'blueprint'
METHOD_HIERARCHICAL = 'hierarchical'

MODE_DEFAULT = 'default'
MODE_CLUSTER = 'cluster'
MODE_FILTERED = 'filtered'

ALLOWED_MODES = {
    METHOD_BLUEPRINT: {MODE_DEFAULT, MODE_CLUSTER},
    METHOD_HIERARCHICAL: {MODE_DEFAULT, MODE_FILTERED},
}

def validate_method_mode(method: str, mode: str):
    if method not in ALLOWED_MODES:
        allowed_methods = ', '.join(ALLOWED_MODES.keys())
        raise ValueError(f'Unknown method `{method}`. Allowed: {allowed_methods}')

    if mode not in ALLOWED_MODES[method]:
        allowed_modes = ', '.join(sorted(ALLOWED_MODES[method]))
        raise ValueError(f'Unknown mode `{mode}` for method `{method}`. Allowed: {allowed_modes}')


class Summarisation:
    '''Основной класс для запуска бенчмарка'''
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
        self.hierarchical = Hierarchical(self.client, self.device, self.encoder, mode='default', think_pass=self.think_pass)

    def prepare_enviroment(self):
        try:
            with open('collection.json', 'r', encoding='utf-8') as f:
                self.collection = json.load(f)
        except:
            raise ValueError('No file with annotations and book texts!')

    async def run_method(self, text: str, method: str = '', mode: str = '', initial_word_limit: int = 500):
        validate_method_mode(method, mode)

        chunks = chunk_text(text)

        if method == METHOD_BLUEPRINT:
            return await self.blueprint.run(chunks=chunks, initial_word_limit=initial_word_limit, mode=mode)

        return await self.hierarchical.run(chunks, initial_word_limit=initial_word_limit, mode=mode)

    def evaluate_annotation(self, ref_annotation, gen_annotation):
        return self.evaluater.evaluate_annotation(ref_annotation, gen_annotation)

    def append_to_json(self, record: dict, output_path):
        line = json.dumps(record, ensure_ascii=False) + '\n'
        with output_path.open('a', encoding='utf-8') as f:
            f.write(line)
            f.flush()

    async def run_benchmark_one_method(
        self, 
        is_evalutation_needed: bool = False,
        number_of_books: int = 3, 
        method: str = 'hierarchical', 
        mode: str = 'default', 
        initial_word_limit: int = 500,
        cap_chars: int = 80000,
        output_dir: str = 'output',
        output_name: str = 'benchmark_results.jsonl'
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / output_name
        
        rows = []
        for idx, item in enumerate(self.collection[:number_of_books]):
            text = '\n'.join(item['text'])
            if cap_chars != -1 and len(text) > cap_chars:
                continue

            start_timer = time.perf_counter()
            summary = await self.run_method(text=text, method=method, mode=mode, initial_word_limit=initial_word_limit)
            end_timer = time.perf_counter()
            runtime = end_timer - start_timer

            generated_annotation = summary
            gold_annotation = item['annotation']
            
            if is_evalutation_needed:
                bertscore, rouge = self.evaluate_annotation(ref_annotation=gold_annotation, gen_annotation=generated_annotation) 
    
                record = {
                    'book_idx': idx,
                    'book_title': item['title'],
                    'book_author': item['author'],
                    'book_genre': item['genre'],
                    'method': method,
                    'mode': mode,
                    'initial_word_limit': initial_word_limit,
                    'text_len (words)': len(text.split()),
                    'annotation_len (words)': len(generated_annotation.split()),
                    'runtime_sec': round(runtime, 4),
                    'bertscore_p': float(bertscore[0]),
                    'bertscore_r': float(bertscore[1]),
                    'bertscore_f': float(bertscore[2]),
                    'rougeL': float(rouge),
                }
    
                rows.append(record)
    
                self.append_to_json(record, output_path=output_path)
            else: 
                record = {
                    'book_idx': idx,
                    'book_title': item['title'],
                    'book_author': item['author'],
                    'book_genre': item['genre'],
                    'method': method,
                    'mode': mode,
                    'initial_word_limit': initial_word_limit,
                    'text_len (words)': len(text.split()),
                    'annotation_len (words)': len(generated_annotation.split()),
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