"""Command-line evaluation entry point."""
import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import build_model, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/test_asycmst_bus_ceus.yaml')
    parser.add_argument('--checkpoint', help='Overrides model.load_path')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--max-batches', type=int, help='Limit batches for a smoke test')
    parser.add_argument('--batch-size', type=int, help='Override params.batch_size')
    parser.add_argument('--num-workers', type=int, help='Override data.num_workers')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if args.batch_size is not None:
        config['params']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    checkpoint_path = args.checkpoint or config['model'].get('load_path')
    if not checkpoint_path:
        raise ValueError('Set model.load_path or pass --checkpoint')
    device = torch.device(config['device'].get('single', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('model', checkpoint))
    model.eval()
    dataset = make_dataset(config, 'test', 'eval')
    loader = DataLoader(dataset, batch_size=config['params']['batch_size'], shuffle=False,
                        num_workers=config['data'].get('num_workers', 4))
    criterion = nn.CrossEntropyLoss()
    labels, predictions, scores, total_loss = [], [], [], 0.0
    with torch.no_grad():
        sample_count = 0
        for batch_index, batch in enumerate(tqdm(loader, desc='test')):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            bus, ceus, target = (batch[key].to(device) for key in ('segment_bus', 'segment_ceus', 'label'))
            logits = model(bus, ceus)
            probability = logits.softmax(1)
            total_loss += criterion(logits, target).item() * target.size(0)
            sample_count += target.size(0)
            labels.extend(target.cpu().tolist()); predictions.extend(probability.argmax(1).cpu().tolist())
            scores.extend(probability[:, 1].cpu().tolist())
    if not sample_count:
        raise ValueError('No batches were processed; --max-batches must be greater than zero')
    metrics = {'loss': total_loss / sample_count, 'accuracy': accuracy_score(labels, predictions),
               'precision': precision_score(labels, predictions, zero_division=0),
               'recall': recall_score(labels, predictions, zero_division=0),
               'f1': f1_score(labels, predictions, zero_division=0),
               'auc': roc_auc_score(labels, scores) if len(set(labels)) > 1 else None,
               'confusion_matrix': confusion_matrix(
                   labels, predictions, labels=list(range(config['data']['num_class']))).tolist()}
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame({'score': scores, 'pred': predictions, 'label': labels}).to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
