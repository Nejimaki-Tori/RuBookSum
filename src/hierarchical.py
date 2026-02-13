from utils import AsyncList, extract_response, _complete
import torch
import gc
from prompts import CHUNK_SUMMARY_PROMPT, SUMMARY_MERGE_WITH_CONTEXT_PROMPT, SUMMARY_MERGE_NO_CONTEXT_PROMPT

class Hierarchical:
    def __init__(
        self, 
        client, 
        device, 
        encoder, 
        mode: str = 'default',
        think_pass: str = ''
    ):
        self.client = client
        self.device = device
        self.encoder = encoder
        self.think_pass = think_pass
        if mode not in ('default', 'filtered'):
            raise ValueError('Wrong mode for Hierarchical! Choose either `default` or `filtered`.')
        self.mode = mode

    def clean_memory(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        
    def filter_near_duplicates(self, summaries: list[str], th: float = 0.85):
        "удаление излишних текстов - 'воды'"
        n = len(summaries)
        
        if n <= 1:
            return summaries
            
        embs = torch.from_numpy(
            self.encoder.encode(
                summaries, 
                batch_size=16, 
                normalize_embeddings=True, 
                device=self.device
            )
        ).to(self.device)
        sim_matrix = embs @ embs.T
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=0).to(self.device)
        masked_sim = sim_matrix.masked_fill(mask, -1)
        max_sim_row, _ = torch.max(masked_sim, dim=1)
        keep_mask = max_sim_row < th
        keep_mask[0] = True
        
        valid_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        valid_summaries = [summaries[i] for i in valid_indices]
        return valid_summaries
        
    async def summarize_chunk(self, chunk: str, word_limit: int = 500):
        myprompt = CHUNK_SUMMARY_PROMPT.format(chunk=chunk, word_limit=word_limit) + self.think_pass
        return await _complete(client=self.client, prompt=myprompt, max_tokens=2048)
    
    async def merge_summaries(
        self, 
        summaries: list[str], 
        word_limit: int = 500, 
        use_context: bool = False, 
        previous_summary: str = ''
    ):
        combined_summary = " ".join(summaries)
    
        if len(combined_summary.split()) > word_limit:
            combined_summary = await self.summarize_chunk(combined_summary, word_limit)
    
        if use_context:
            myprompt = SUMMARY_MERGE_WITH_CONTEXT_PROMPT.format(
                previous_summary=previous_summary, 
                combined_summary=combined_summary, 
                word_limit=word_limit
            ) + self.think_pass
        else:
            myprompt = SUMMARY_MERGE_NO_CONTEXT_PROMPT.format(
                combined_summary=combined_summary, 
                word_limit=word_limit
            ) + self.think_pass

        return await _complete(client=self.client, prompt=myprompt, max_tokens=2048)
    
    async def merge_group(self, group1: list[str], group2: list[str], word_limit: int = 500):
        temp_summary = await self.merge_summaries(group1, word_limit=word_limit)
        return await self.merge_summaries(group2, word_limit=word_limit, use_context=True, previous_summary=temp_summary)
        
    async def hierarchical_summary(self, chunks: list[str], initial_word_limit: int = 500):
        if not chunks:
            raise ValueError("`chunks` должен содержать хотя бы один элемент!")
            
        rest_chunks = self.filter_near_duplicates(chunks) if self.mode == 'filtered' else chunks
        if self.mode == 'filtered':
            self.clean_memory()
        results = AsyncList()
    
        for chunk in rest_chunks:
            results.append(self.summarize_chunk(chunk, initial_word_limit))
    
        await results.complete_couroutines(batch_size=40)
        summaries = await results.to_list()
        current_level_summaries = summaries
        current_word_limit = initial_word_limit
    
        if len(current_level_summaries) == 1:
            return current_level_summaries[0]
            
        if len(current_level_summaries) == 2:
            return await self.merge_summaries(current_level_summaries, current_word_limit)
        
        count = 0
        while len(current_level_summaries) > 2:
            count += 1
            i = 0
            tasks = AsyncList()
            while i < len(current_level_summaries):
                if i + 2 < len(current_level_summaries):
                    group1 = current_level_summaries[i: i + 3]
                    if i + 5 < len(current_level_summaries):
                        group2 = current_level_summaries[i + 3: i + 6]
                        tasks.append(self.merge_group(group1, group2, current_word_limit))
                        i += 6
                    else:
                        tasks.append(self.merge_summaries(group1, current_word_limit))
                        i += 3
                else:
                    tasks.append(current_level_summaries[i])
                    i += 1
            await tasks.complete_couroutines(batch_size=40)
            next_level_summaries = await tasks.to_list()

            current_level_summaries = self.filter_near_duplicates(next_level_summaries) if self.mode == 'filtered' else next_level_summaries
            if self.mode == 'filtered':
                self.clean_memory()
    
        if len(current_level_summaries) == 1:
            return current_level_summaries[0]
            
        return await self.merge_summaries(current_level_summaries, current_word_limit)
        
    async def run(self, chunks: list[str], initial_word_limit: int = 500, mode: str = 'default'):
        self.mode = mode
        s = await self.hierarchical_summary(chunks, initial_word_limit)
        self.clean_memory()
        return s