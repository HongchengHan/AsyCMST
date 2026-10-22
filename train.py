"""Command-line training entry point for AsyCMST and TSNet."""
import argparse
import datetime
import os

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.asycmst import AsyCMST, dice_loss
from models.tsnet import tsnet_rx18, tsnet_rx50
from utils.dataloader_cls import BUS_CEUS_Classification_Images


def build_model(config):
    data, model = config['data'], config['model']
    common = dict(num_classes=data['num_class'], sample_size=data['size'], sample_duration=data['num_frm'])
    if model['name'] == 'asycmst_r18_384_6_6':
        return AsyCMST(**common, embed_dim=384, num_layers=6, num_heads=6,
                       use_mask=data.get('use_mask', False), pretrained=model.get('pretrain', False))
    if model['name'] == 'tsnet_rx18':
        return tsnet_rx18(**common)
    if model['name'] == 'tsnet_rx50':
        return tsnet_rx50(**common)
    raise ValueError(f"Unknown model: {model['name']}")


def make_dataset(config, split, mode):
    data = config['data']
    return BUS_CEUS_Classification_Images(
        data['root_dir'], data[f'{split}_label'], data['size'], data['num_frm'], mode=mode,
        augment=data.get('augment', False), aug=data.get('aug'), use_mask=data.get('use_mask', False),
        mask_column=data.get('mask_column', 'mask_dir'), mask_root=data.get('mask_root', 'masks'))


def metric_summary(labels, predictions, scores):
    result = {'acc': accuracy_score(labels, predictions), 'f1': f1_score(labels, predictions, zero_division=0)}
    result['auc'] = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    return result


def run_epoch(model, loader, criterion, device, optimizer=None, fss_weight=0.4, seg_weight=0.4,
              max_batches=None):
    training = optimizer is not None
    model.train(training)
    losses, labels, predictions, scores = [], [], [], []
    sample_count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc='train' if training else 'valid', leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        bus, ceus, target = (batch[key].to(device) for key in ('segment_bus', 'segment_ceus', 'label'))
        with torch.set_grad_enabled(training):
            if isinstance(model, AsyCMST):
                output = model(bus, ceus, return_aux=True)
                logits = output['logits']
                loss = criterion(logits, target) + fss_weight * model.frame_sorting.loss(output['frame_scores'])
                if model.use_mask:
                    loss = loss + seg_weight * dice_loss(output['mask_logits'], batch['mask'].to(device))
            else:
                logits, loss = model(bus, ceus), None
                loss = criterion(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        probability = logits.softmax(1)
        losses.append(loss.item() * target.size(0))
        sample_count += target.size(0)
        labels.extend(target.cpu().tolist())
        predictions.extend(probability.argmax(1).cpu().tolist())
        scores.extend(probability[:, 1].detach().cpu().tolist())
    if not sample_count:
        raise ValueError('No batches were processed; max_batches must be greater than zero')
    return sum(losses) / sample_count, metric_summary(labels, predictions, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_asycmst_bus_ceus.yaml')
    parser.add_argument('--epochs', type=int, help='Override params.epoch')
    parser.add_argument('--batch-size', type=int, help='Override params.batch_size')
    parser.add_argument('--num-workers', type=int, help='Override data.num_workers')
    parser.add_argument('--max-train-batches', type=int, help='Limit batches for a smoke test')
    parser.add_argument('--max-valid-batches', type=int, help='Limit batches for a smoke test')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if args.epochs is not None:
        config['params']['epoch'] = args.epochs
    if args.batch_size is not None:
        config['params']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    device = torch.device(config['device'].get('single', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    params = config['params']
    optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], momentum=params.get('momentum', .9),
                                    weight_decay=params.get('w_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.get('factor', .5),
                                                           patience=params.get('patience', 3))
    weights = torch.tensor(params.get('ECLossWeight', [1] * config['data']['num_class']), device=device)
    criterion = nn.CrossEntropyLoss(weights)
    workers = config['data'].get('num_workers', 4)
    train_loader = DataLoader(make_dataset(config, 'train', 'train'), batch_size=params['batch_size'],
                              shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(make_dataset(config, 'valid', 'eval'), batch_size=params['batch_size'],
                              num_workers=workers, pin_memory=torch.cuda.is_available())
    stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_root = config.get('save', {}).get('path')
    if not save_root or str(save_root).lower() == 'none':
        save_root = 'runs'
    run_dir = os.path.join(save_root, config['task'], f"{stamp}_{config['model']['name']}")
    checkpoint_dir = os.path.join(run_dir, 'ckpt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer, best = SummaryWriter(run_dir), -1.0
    for epoch in range(params['epoch']):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer,
                                               params.get('fss_weight', .4), params.get('seg_weight', .4),
                                               args.max_train_batches)
        valid_loss, valid_metrics = run_epoch(model, valid_loader, criterion, device, None,
                                               params.get('fss_weight', .4), params.get('seg_weight', .4),
                                               args.max_valid_batches)
        scheduler.step(valid_loss)
        for split, loss, metrics in [('train', train_loss, train_metrics), ('valid', valid_loss, valid_metrics)]:
            writer.add_scalar(f'loss/{split}', loss, epoch)
            for name, value in metrics.items(): writer.add_scalar(f'{name}/{split}', value, epoch)
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f} acc={train_metrics['acc']:.4f}; "
              f"valid loss={valid_loss:.4f} acc={valid_metrics['acc']:.4f} auc={valid_metrics['auc']:.4f}")
        if valid_metrics['acc'] >= best:
            best = valid_metrics['acc']
            torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch},
                       os.path.join(checkpoint_dir, 'best.pth'))
    writer.close()


if __name__ == '__main__':
    main()
