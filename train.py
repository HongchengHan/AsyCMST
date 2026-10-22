"""Command-line training entry point for AsyCMST and TSNet."""
import argparse
import datetime
import os

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.asycmst import AsyCMST, dice_loss
from models.baseline import DualResNet18Baseline
from models.transformer_baseline import DualTransformerBaseline
from models.tsnet import tsnet_rx18, tsnet_rx50
from utils.dataloader_cls import BUS_CEUS_Classification_Images


def build_model(config):
    data, model = config['data'], config['model']
    common = dict(num_classes=data['num_class'], sample_size=data['size'], sample_duration=data['num_frm'])
    if model['name'] == 'asycmst_r18_384_6_6':
        return AsyCMST(**common, embed_dim=384, num_layers=6, num_heads=6,
                       use_mask=data.get('use_mask', False), pretrained=model.get('pretrain', False),
                       cross_gate_init=model.get('cross_gate_init', 0.1),
                       cross_attention_start_layer=model.get('cross_attention_start_layer', 3))
    if model['name'] == 'dual_resnet18_baseline':
        return DualResNet18Baseline(
            **common, pretrained=model.get('pretrain', True), dropout=model.get('dropout', 0.3))
    if model['name'] == 'dual_transformer_baseline':
        return DualTransformerBaseline(
            **common, embed_dim=model.get('embed_dim', 384),
            num_layers=model.get('num_layers', 6), num_heads=model.get('num_heads', 6),
            pretrained=model.get('pretrain', True), dropout=model.get('dropout', 0.1))
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
        mask_column=data.get('mask_column', 'mask_dir'), mask_root=data.get('mask_root', 'masks'),
        normalize=data.get('normalize', True))


def metric_summary(labels, predictions, scores):
    result = {'acc': accuracy_score(labels, predictions),
              'balanced_acc': balanced_accuracy_score(labels, predictions),
              'f1': f1_score(labels, predictions, zero_division=0)}
    result['auc'] = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    for class_index in range(2):
        result[f'pred_class_{class_index}_ratio'] = predictions.count(class_index) / len(predictions)
    return result


def build_optimizer(model, params):
    name = params.get('optimizer', 'adamw').lower()
    common = {'lr': params['lr'], 'weight_decay': params.get('w_decay', 1e-4)}
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **common)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), **common)
    if name == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(), momentum=params.get('momentum', .9), **common)
    raise ValueError(f"Unsupported optimizer: {name}. Choose adamw, adam, or rmsprop.")


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
    optimizer = build_optimizer(model, params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.get('factor', .5),
                                                           patience=params.get('patience', 3))
    workers = config['data'].get('num_workers', 4)
    train_dataset = make_dataset(config, 'train', 'train')
    valid_dataset = make_dataset(config, 'valid', 'eval')
    configured_weights = params.get('ECLossWeight', 'auto')
    if configured_weights == 'auto':
        counts = torch.bincount(torch.as_tensor(train_dataset.labels, dtype=torch.long),
                                minlength=config['data']['num_class']).float()
        weights = counts.sum() / (len(counts) * counts.clamp_min(1))
    else:
        weights = torch.tensor(configured_weights, dtype=torch.float)
    weights = weights.to(device)
    print(f'Cross-entropy class weights: {weights.cpu().tolist()}')
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'],
                              shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'],
                              num_workers=workers, pin_memory=torch.cuda.is_available())
    stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_root = config.get('save', {}).get('path')
    if not save_root or str(save_root).lower() == 'none':
        save_root = 'runs'
    run_dir = os.path.join(save_root, config['task'], f"{stamp}_{config['model']['name']}")
    checkpoint_dir = os.path.join(run_dir, 'ckpt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer, best_loss, best_auc = SummaryWriter(run_dir), float('inf'), float('-inf')
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
              f"valid loss={valid_loss:.4f} acc={valid_metrics['acc']:.4f} auc={valid_metrics['auc']:.4f}; "
              f"valid pred(0/1)={valid_metrics.get('pred_class_0_ratio', 0):.3f}/"
              f"{valid_metrics.get('pred_class_1_ratio', 0):.3f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                        'valid_loss': valid_loss, 'valid_metrics': valid_metrics},
                       os.path.join(checkpoint_dir, 'best_loss.pth'))
        if valid_metrics['auc'] > best_auc:
            best_auc = valid_metrics['auc']
            torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                        'valid_loss': valid_loss, 'valid_metrics': valid_metrics},
                       os.path.join(checkpoint_dir, 'best_auc.pth'))
    writer.close()


if __name__ == '__main__':
    main()
