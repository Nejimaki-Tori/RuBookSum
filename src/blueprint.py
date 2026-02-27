from utils import AsyncList, extract_response, _complete
from hierarchical import Hierarchical
from sklearn.cluster import KMeans
import math
import random
import numpy as np
from prompts import BLUEPRINT_QUESTIONS_PROMPT, BLUEPRINT_ANSWER_PROMPT, GENERALISE_QUESTIONS_PROMPT, SUMMARIZE_BLUEPRINT_PROMPT, SUMMARIZE_BLUEPRINT_NO_ANSWERS_PROMPT

RANDOM_SEED = 42


class Blueprint(Hierarchical):
    def __init__(
        self, 
        client, 
        device, 
        encoder, 
        mode: str = 'default', 
        think_pass: str = '',
        concurrency: int = 40
    ):
        super().__init__(client, device, encoder, mode='default', think_pass=think_pass)
        if mode not in ('default', 'cluster'):
            raise ValueError('Wrong mode for Blueprint! Choose either `default` or `cluster`.')

        self.concurrency = concurrency
        self.mode = mode
    
    async def summarize_with_blueprint(self, chunk, blueprint):
        if self.mode == 'default':
            myprompt = SUMMARIZE_BLUEPRINT_PROMPT.format(blueprint=blueprint, chunk=chunk) + self.think_pass
        else:
            myprompt = SUMMARIZE_BLUEPRINT_NO_ANSWERS_PROMPT.format(blueprint=blueprint, chunk=chunk) + self.think_pass
            
        return await _complete(client=self.client, prompt=myprompt, max_tokens=2048)

    async def generate_questions_chunk(self, chunk: str):
        myprompt = BLUEPRINT_QUESTIONS_PROMPT.format(chunk=chunk) + self.think_pass
        raw_questions = await _complete(client=self.client, prompt=myprompt, max_tokens=1024)
        
        questions = [q.strip() for q in raw_questions.split('\n') if q.strip()]
        return questions

    async def get_answer(self, chunk: str, question: str):
        myprompt = BLUEPRINT_ANSWER_PROMPT.format(chunk=chunk, question=question) + self.think_pass
        return await _complete(client=self.client, prompt=myprompt, max_tokens=256)
        
    async def generate_answers(self, chunk: str, questions: list[str]) -> list[str]:
        answers = AsyncList()

        for question in questions:
            answers.append(self.get_answer(chunk, question))

        await answers.complete_couroutines(batch_size=self.concurrency)
        answers = await answers.to_list()

        return answers

    def cluster_questions(self, questions: list[str]):
        '''Кластеризация вопросов'''
        embs = self.encoder.encode(questions, batch_size=16, normalize_embeddings=True, device=self.device)
        
        n_clusters = min(max(2, int(math.sqrt(len(questions)))), 15) # N_clusters <= 15
        
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

        labels = kmeans.fit_predict(embs)

        clusters = {}
        for label, question in zip(labels, questions):
            clusters.setdefault(label, []).append(question)

        return clusters

    def merge_answers_and_questions(self, questions: list[str], answers: list[str]):
        result = []
        for q, a in zip(questions, answers):
            tmp = '\n'.join([question + ' -- ' + answer for question, answer in zip(q, a)])
            result.append(tmp)
            
        return result

    async def generalize_questions(self, questions_cl: list[str], question_limit: int = 10):
        k = min(len(questions_cl), question_limit)
        sampled_questions = random.sample(questions_cl, k=k)
        
        questions_str = '\n'.join(f'- {q}' for q in sampled_questions)
        myprompt = GENERALISE_QUESTIONS_PROMPT.format(questions=questions_str) + self.think_pass
        return await _complete(client=self.client, prompt=myprompt, max_tokens=128)

    async def generate_blueprint(self, chunks: list[str]) -> list[str]:
        if not chunks:
            raise ValueError('Empty chunks!')

        questions_set = AsyncList()
        for chunk in chunks:
            questions_set.append(self.generate_questions_chunk(chunk))
        
        await questions_set.complete_couroutines(batch_size=self.concurrency)
        questions_set = await questions_set.to_list()

        if self.mode == 'cluster':
            all_questions_list = [q.strip() for qs in questions_set for q in qs if q.strip()]
            if not all_questions_list:
                raise ValueError('No questions have been generated!')
                
            question_clusters = self.cluster_questions(all_questions_list)
            generalized = AsyncList()
            for questions in question_clusters.values():
                generalized.append(self.generalize_questions(questions, question_limit=10))
            await generalized.complete_couroutines(batch_size=self.concurrency)
            generalized_questions = await generalized.to_list()
            
            blueprint = '\n'.join(q.strip() for q in generalized_questions if q.strip())
            return [blueprint] * len(chunks)

        answers = []
        for chunk, questions_per_chunk in zip(chunks, questions_set):
            answers.append(await self.generate_answers(chunk, questions_per_chunk))
        blueprint = self.merge_answers_and_questions(questions_set, answers)
        return blueprint

    async def merge_pair(self, sum1: str, sum2: str, word_limit: int, blueprint: str = ''): # для сжатия аннотаций
        if not sum2:
            return sum1
        combo = f"{sum1} {sum2}".strip()
        if len(combo.split()) > word_limit:
            if self.mode == 'default':
                bp = (await self.generate_blueprint([combo]))[0]
            else:
                bp = blueprint
            combo = await self.summarize_with_blueprint(combo, bp)
        return combo
    
    async def text_blueprint_summary(self, chunks: list[str], word_limit=500):
        blueprint = await self.generate_blueprint(chunks)
        summaries_list = AsyncList()

        for chunk, bp in zip(chunks, blueprint):
            summaries_list.append(self.summarize_with_blueprint(chunk, bp))
                
        await summaries_list.complete_couroutines(batch_size=self.concurrency)
        summaries = await summaries_list.to_list()
        while len(summaries) > 1:
            tasks = AsyncList()
            i = 0
    
            while i < len(summaries):
                sum1 = summaries[i]
                sum2 = summaries[i + 1] if i + 1 < len(summaries) else None
                tasks.append(self.merge_pair(sum1, sum2, word_limit, blueprint if self.mode == 'cluster' else None))
                i = i + 2 if i + 1 < len(summaries) else i + 1
                
            await tasks.complete_couroutines(batch_size=self.concurrency)
            summaries = await tasks.to_list()
            blueprint = await self.generate_blueprint(summaries) if self.mode == 'cluster' else None
    
        final_summary = summaries[0].strip()
        if len(final_summary.split()) > word_limit:
            if self.mode == 'default':
                bp = (await self.generate_blueprint([final_summary]))[0]
            else:
                bp = blueprint
            final_summary = await self.summarize_with_blueprint(final_summary, bp)
    
        return final_summary

    async def run(self, chunks: list[str], initial_word_limit: int = 500, mode: str = 'default'):
        random.seed(RANDOM_SEED)
        self.mode = mode
        
        s = await self.text_blueprint_summary(chunks, initial_word_limit)
        self.clean_memory()
        return s
