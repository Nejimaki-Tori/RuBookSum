import sys
import csv
import json
import torch
import asyncio
import logging
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

repo_root = Path(__file__).resolve().parent
src_dir = repo_root / 'src'

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from methods import Summarisation


def resolve_device(device_arg: str):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_encoder(encoder_name: str, device):
    logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.ERROR)
    return SentenceTransformer(encoder_name).to(device)

def save_config(path, config: dict):
    with path.open('w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def save_metrics_json(path, metrics: dict):
    with path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def flatten_metrics(metrics: dict) -> dict:
    flat = {
        'model_name': metrics['model_name'],
        'number_of_books': metrics['number_of_books'],
    }

    for key in ('bertscore_p', 'bertscore_r', 'bertscore_f', 'rougeL', 'runtime'):
        metric = metrics.get(key)
        flat[f'{key}_mean'] = metric.get('mean')
        flat[f'{key}_ci_low'] = metric.get('ci_low')
        flat[f'{key}_ci_high'] = metric.get('ci_high')

    return flat

def save_metrics_csv(path, metrics: dict):
    row = flatten_metrics(metrics)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='RuBookSum', description='Run RuBookSum benchmark')

    parser.add_argument('--api', required=True, help='LLM API base URL')
    parser.add_argument('--key', required=True, help='LLM API key')
    parser.add_argument('--model-name', required=True, help='Model name for API calls')
    parser.add_argument('--concurrency', type=int, required=True, help='Async concurrency (e.g. batch size)')
    parser.add_argument('--output-dir', required=True, help='Directory for outputs')

    parser.add_argument('--number-of-books', type=int, default=5, help='Number of books used for evaluation')
    parser.add_argument('--encoder-name', default='deepvk/USER-bge-m3', help='Encoder for embeddings')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    parser.add_argument('--method', type=str, default='hierarchical', choices=['hierarchical', 'blueprint'])
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'cluster', 'filtered'])

    parser.add_argument('--initial-word-limit', type=int, default=500)
    parser.add_argument('--cap-chars', type=int, default=80000)

    return parser

async def run(args):
    device = resolve_device(args.device)
    encoder = build_encoder(args.encoder_name, device)

    model_safe_name = args.model_name.replace('/', '_').replace(' ', '_')
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / model_safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    save_config(model_dir / 'config.json', {
        'api': args.api,
        'model_name': args.model_name,
        'concurrency': args.concurrency,
        'number_of_books': args.number_of_books,
        'encoder_name': args.encoder_name,
        'device': str(device),
        'method': args.method,
        'mode': args.mode,
    })

    bench = Summarisation(
        url=args.api,
        key=args.key,
        model_name=args.model_name,
        device=device,
        encoder=encoder,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
    )

    bench.prepare_environment()

    metrics = await bench.run_benchmark_one_method(
        is_evaluation_needed=True,
        number_of_books=args.number_of_books,
        method=args.method,
        mode=args.mode,
        initial_word_limit=args.initial_word_limit,
        cap_chars=args.cap_chars
    )

    metrics.update({
        'model_name': args.model_name,
        'number_of_books': args.number_of_books,
    })

    save_metrics_json(model_dir / 'metrics.json', metrics)
    save_metrics_csv(model_dir / 'metrics.csv', metrics)
    
    return metrics

def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))

if __name__ == '__main__':
    main()
