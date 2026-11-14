#!/usr/bin/env python3
"""BITS F445 PA2: controlled MLP experiments and final-model training.

Reproduction (Python with torch, numpy, pandas and matplotlib already installed):
    python 2024ADPS0831G.py --data-dir PATH_TO_PA2_SUPPORTING_FILES --stage required
    python 2024ADPS0831G.py --data-dir PATH_TO_PA2_SUPPORTING_FILES --stage search
    python 2024ADPS0831G.py --data-dir PATH_TO_PA2_SUPPORTING_FILES --stage refine
    python 2024ADPS0831G.py --data-dir PATH_TO_PA2_SUPPORTING_FILES --stage diagnostic
    python 2024ADPS0831G.py --data-dir PATH_TO_PA2_SUPPORTING_FILES --stage final

The directory must include training.csv, validation.csv and save_model_json.py.
Run the stages in that order. --stage all runs all stages. By default, generated
histories, figures, selection evidence and the JSON go to ./pa2_results.
The required comparison runs always use 120 epochs; the final retraining uses
the validation-selected checkpoint epoch (never more than 120).

AI acknowledgement: OpenAI Codex assisted with the code, experiment execution,
analysis and report. All reported numbers are computed from the supplied data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 831031
BITS_ID = '2024ADPS0831G'
FEATURES = [f'x{i}' for i in range(1, 25)]
TARGET = 'condition'
BATCH_SIZE = 64
MAX_EPOCHS = 120


@dataclass(frozen=True)
class Config:
    hidden: tuple[int, ...] = (32,)
    activation: str = 'relu'
    initialization: str = 'kaiming'
    loss: str = 'ce'
    lr: float = 0.03
    weight_decay: float = 0.0


class MLP(nn.Module):
    def __init__(self, hidden, activation):
        super().__init__()
        activations = {'identity': nn.Identity, 'sigmoid': nn.Sigmoid,
                       'tanh': nn.Tanh, 'relu': nn.ReLU}
        dims = [24, *hidden, 6]
        layers = []
        for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if i < len(dims) - 2:
                layers.append(activations[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_model(config):
    torch.manual_seed(SEED)
    model = MLP(config.hidden, config.activation)
    # Reset before explicit initialization; same-shaped comparison models get
    # identical random draws regardless of their activation or loss function.
    torch.manual_seed(SEED)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if config.initialization == 'small':
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            elif config.initialization == 'large':
                nn.init.normal_(layer.weight, mean=0, std=1.0)
            elif config.initialization == 'nielsen':
                nn.init.normal_(layer.weight, mean=0, std=1 / math.sqrt(layer.in_features))
            elif config.initialization == 'kaiming':
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            else:
                raise ValueError(config.initialization)
            nn.init.zeros_(layer.bias)
    return model


def objective(logits, labels, loss_name):
    if loss_name == 'ce':
        return nn.functional.cross_entropy(logits, labels)
    if loss_name == 'quadratic':
        target = nn.functional.one_hot(labels, num_classes=6).to(logits.dtype)
        # Nielsen: 1/(2n) sum_examples sum_classes (probability-target)^2.
        return 0.5 * (logits.softmax(dim=1) - target).square().sum(dim=1).mean()
    raise ValueError(loss_name)


def metrics(model, x, y, loss_name):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        loss = objective(logits, y, loss_name).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy


def weights_l2(model):
    # Euclidean norm after concatenating all weight matrices; excludes biases.
    return math.sqrt(sum(m.weight.detach().square().sum().item()
                         for m in model.modules() if isinstance(m, nn.Linear)))


def read_data(data_dir, out_dir):
    frames = [pd.read_csv(data_dir / name) for name in ('training.csv', 'validation.csv')]
    arrays = []
    for frame in frames:
        x = frame[FEATURES].to_numpy(dtype=np.float64)
        y_num = frame[TARGET].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y_num).all():
            raise ValueError('Non-finite input or target')
        if not np.equal(y_num, np.floor(y_num)).all() or not ((y_num >= 0) & (y_num < 6)).all():
            raise ValueError('Expected integer labels 0 through 5')
        arrays.append((x, y_num.astype(np.int64)))
    (x_train, y_train), (x_val, y_val) = arrays
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0, ddof=0)
    assert (feature_std > 1e-12).all()
    # Immutable separate names prevent the PA1 loss-history / scaling collision.
    feature_mean.setflags(write=False)
    feature_std.setflags(write=False)
    z_train = ((x_train - feature_mean) / feature_std).astype(np.float32)
    z_val = ((x_val - feature_mean) / feature_std).astype(np.float32)
    assert np.allclose(z_train.mean(0), 0, atol=1e-6)
    assert np.allclose(z_train.std(0, ddof=0), 1, atol=1e-6)
    np.savez(out_dir / 'preprocessing.npz', feature_mean=feature_mean, feature_std=feature_std)
    audit = {
        'seed': SEED, 'train_shape': list(x_train.shape), 'validation_shape': list(x_val.shape),
        'train_counts': np.bincount(y_train, minlength=6).tolist(),
        'validation_counts': np.bincount(y_val, minlength=6).tolist(),
        'train_duplicate_rows': int(frames[0].duplicated().sum()),
        'train_validation_identical_feature_rows': len(set(map(tuple, x_train)) & set(map(tuple, x_val))),
        'feature_mean': feature_mean.tolist(), 'feature_std': feature_std.tolist(),
        'sha256': {name: hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
                   for name in ('training.csv', 'validation.csv')},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__, 'threads': torch.get_num_threads()}}
    (out_dir / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    return tuple(torch.from_numpy(a) for a in (z_train, y_train, z_val, y_val))


def checkpoint_key(row):
    # Highest accuracy, then lowest held-out loss, then earliest epoch.
    return (-row['val_accuracy'], row['val_loss'], row['epoch'])


def train_run(name, config, data, out_dir, epochs=MAX_EPOCHS, use_cache=True):
    history_path = out_dir / 'histories' / f'{name}.csv'
    metadata_path = out_dir / 'histories' / f'{name}.json'
    best_path = out_dir / 'checkpoints' / f'{name}.pt'
    final_path = out_dir / 'checkpoints' / f'{name}_last.pt'
    cfg_dict = json.loads(json.dumps(asdict(config)))
    if use_cache and history_path.exists() and metadata_path.exists() and best_path.exists() and final_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta['config'] == cfg_dict and meta['epochs_requested'] == epochs:
            return pd.read_csv(history_path), meta
    x_train, y_train, x_val, y_val = data
    model = create_model(config)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0,
                                weight_decay=config.weight_decay)
    history, best = [], None
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(xb), yb, config.loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in {name}, epoch {epoch}')
            loss.backward()
            optimizer.step()
        train_loss, train_acc = metrics(model, x_train, y_train, config.loss)
        val_loss, val_acc = metrics(model, x_val, y_val, config.loss)
        row = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
               'val_loss': val_loss, 'val_accuracy': val_acc, 'weight_l2': weights_l2(model)}
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError(f'Non-finite metrics in {name}, epoch {epoch}')
        history.append(row)
        if best is None or checkpoint_key(row) < checkpoint_key(best):
            best = row.copy()
            torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), final_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    meta = {'name': name, 'config': cfg_dict, 'epochs_requested': epochs,
            'seed': SEED, 'batch_size': BATCH_SIZE, 'momentum': 0,
            'weight_decay_includes_bias': True, 'best': best, 'final': history[-1],
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'seconds': time.perf_counter() - started}
    metadata_path.write_text(json.dumps(meta, indent=2))
    if name == 'FINAL':
        print(f'{name}: final combined training accuracy={train_acc:.4%}; '
              f'CE={train_loss:.6f}; {meta["seconds"]:.1f}s',flush=True)
    else:
        print(f'{name}: best val={best["val_accuracy"]:.4%} at {best["epoch"]}; '
              f'last train/val={train_acc:.4%}/{val_acc:.4%}; {meta["seconds"]:.1f}s', flush=True)
    return pd.DataFrame(history), meta


def required_configs():
    configs = {}
    for activation in ('identity', 'sigmoid', 'tanh', 'relu'):
        configs[f'A_{activation}'] = Config(activation=activation, initialization='nielsen', lr=.05)
    for loss in ('ce', 'quadratic'):
        configs[f'B_{loss}'] = Config(loss=loss, lr=.05)
    for init in ('small', 'large', 'nielsen', 'kaiming'):
        configs[f'C_{init}'] = Config(hidden=(64,32), initialization=init)
    for name, hidden in [('tiny',(8,)), ('small',(32,)), ('medium',(64,32)), ('large',(128,128))]:
        configs[f'D_{name}'] = Config(hidden=hidden)
    for wd in (0., .001, .01, .03, .1):
        configs[f'E_wd{wd:g}'] = Config(hidden=(128,128), weight_decay=wd)
    return configs


def plot_comparison(names, labels, output, out_dir, early=False):
    fig, axes = plt.subplots(1,2,figsize=(10.4,3.5))
    for name, label in zip(names, labels):
        h = pd.read_csv(out_dir / 'histories' / f'{name}.csv')
        axes[0].plot(h.epoch, 100*h.val_accuracy, label=label)
        axes[1].plot(h.epoch, h.val_loss, label=label)
    axes[0].set_ylabel('Validation accuracy (%)')
    axes[1].set_ylabel('Validation loss')
    if output.startswith('C_'):
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Validation CE (log scale)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.22); ax.legend(fontsize=8)
        if early:
            ax.set_xlim(1, 20)
    fig.tight_layout()
    fig.savefig(out_dir / 'figures' / output, dpi=190)
    plt.close(fig)


def required_plots(out_dir):
    plot_comparison([f'A_{a}' for a in ('identity','sigmoid','tanh','relu')],
                    ['Identity','Sigmoid','Tanh','ReLU'], 'A_activation.png', out_dir)
    plot_comparison([f'C_{i}' for i in ('small','large','nielsen','kaiming')],
                    ['Small normal','Large normal','Nielsen','Kaiming'], 'C_initialization.png', out_dir)
    plot_comparison([f'C_{i}' for i in ('small','large','nielsen','kaiming')],
                    ['Small normal','Large normal','Nielsen','Kaiming'], 'C_early.png', out_dir, early=True)
    plot_comparison([f'D_{n}' for n in ('tiny','small','medium','large')],
                    ['Tiny: 8','Small: 32','Medium: 64-32','Large: 128-128'], 'D_capacity.png', out_dir)
    plot_comparison([f'E_wd{w:g}' for w in (0.,.001,.01,.03,.1)],
                    ['0','0.001','0.01','0.03','0.1'], 'E_decay.png', out_dir)
    fig, axes = plt.subplots(1,3,figsize=(11,3.3))
    for loss, label in [('ce','Cross-entropy'),('quadratic','Quadratic')]:
        h = pd.read_csv(out_dir / 'histories' / f'B_{loss}.csv')
        axes[0].plot(h.epoch,100*h.val_accuracy,label=label)
        ax = axes[1 if loss=='ce' else 2]
        ax.plot(h.epoch,h.val_loss,color='C0' if loss=='ce' else 'C1')
        ax.set_title(label); ax.set_ylabel('Validation loss')
    axes[0].set_ylabel('Validation accuracy (%)'); axes[0].legend(fontsize=8)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(out_dir/'figures'/'B_loss.png',dpi=190); plt.close(fig)
    h = pd.read_csv(out_dir/'histories'/'D_large.csv')
    fig, axes = plt.subplots(1,2,figsize=(10.4,3.4))
    for prefix, label in [('train','Training'),('val','Validation')]:
        axes[0].plot(h.epoch,100*h[f'{prefix}_accuracy'],label=label)
        axes[1].plot(h.epoch,h[f'{prefix}_loss'],label=label)
    axes[0].set_ylabel('Accuracy (%)'); axes[1].set_ylabel('Cross-entropy')
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.22); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir/'figures'/'D_large_curves.png',dpi=190); plt.close(fig)


def run_required(data, out_dir):
    for name, config in required_configs().items():
        train_run(name, config, data, out_dir)
    # Fairness checks: identical starting parameters for activation and cost comparisons.
    for group in [('A_identity','A_sigmoid','A_tanh','A_relu'),('B_ce','B_quadratic')]:
        states = [create_model(required_configs()[name]).state_dict() for name in group]
        assert all(all(torch.equal(states[0][key], state[key]) for key in states[0]) for state in states[1:])
    for left,right in [('C_kaiming','D_medium'),('D_large','E_wd0')]:
        assert np.allclose(pd.read_csv(out_dir/'histories'/f'{left}.csv'),
                           pd.read_csv(out_dir/'histories'/f'{right}.csv'),atol=1e-9)
    required_plots(out_dir)
    print('Controlled comparisons completed; starting-parameter and repeated-run checks passed.',flush=True)


def search_configs():
    # A small, declared extension of the required experiments. Every candidate
    # stays inside section 10: same optimizer/batch size/seed and <=120 epochs.
    configs = {}
    for activation, init in [('relu','kaiming'),('tanh','nielsen')]:
        for hidden in ((32,), (64,32), (128,128), (128,64,32)):
            for wd in (.001, .01):
                name = f'S_{activation}_{"-".join(map(str,hidden))}_wd{wd:g}'
                configs[name] = Config(hidden=hidden, activation=activation, initialization=init,
                                       lr=.03, weight_decay=wd)
    return configs


def run_search(data, out_dir):
    for name, config in search_configs().items():
        train_run(name, config, data, out_dir)
    print('Search stage complete.',flush=True)


def run_refinement(data, out_dir):
    # Follow up the two leading distinct architectures, without changing seed
    # or using any combined-data training score to rank the alternatives.
    leaders=[]
    for candidate in all_candidates(out_dir):
        if not candidate['name'].startswith(('A_','B_','C_','D_','E_','S_')):
            continue  # Keep this follow-up grid fixed on repeated invocations.
        cfg=candidate['config']
        key=(tuple(cfg['hidden']),cfg['activation'],cfg['initialization'])
        if key not in [k for k,_ in leaders]:
            leaders.append((key,cfg))
        if len(leaders)==2: break
    already={json.dumps(m['config'],sort_keys=True) for m in all_candidates(out_dir)}
    plans=[]
    for _,cfg in leaders:
        for lr,wd in [(.01,.01),(.01,.03),(.05,.01),(.05,.03),(.03,.005),(.03,.02)]:
            plans.append(Config(hidden=tuple(cfg['hidden']),activation=cfg['activation'],
                                initialization=cfg['initialization'],lr=lr,weight_decay=wd))
    # Part C favored Nielsen over Kaiming without decay, so check that
    # initialization in regularized ReLU models and in wider single layers.
    for hidden in ((64,),(128,),(128,128)):
        for wd in (.01,.03):
            plans.append(Config(hidden=hidden,initialization='nielsen',lr=.03,weight_decay=wd))
    for cfg in plans:
        key=json.dumps(json.loads(json.dumps(asdict(cfg))),sort_keys=True)
        if key in already: continue
        name=(f'R_{cfg.activation}_{"-".join(map(str,cfg.hidden))}_{cfg.initialization}'
              f'_lr{cfg.lr:g}_wd{cfg.weight_decay:g}')
        train_run(name,cfg,data,out_dir)
        already.add(key)
    names=[];labels=[];seen=set()
    for item in all_candidates(out_dir):
        key=json.dumps(item['config'],sort_keys=True)
        if key in seen: continue
        seen.add(key)
        c=item['config']
        names.append(item['name'])
        labels.append(f'{"-".join(map(str,c["hidden"]))} {c["initialization"]}, lr {c["lr"]:g}, wd {c["weight_decay"]:g}')
        if len(names)==4: break
    plot_comparison(names,labels,'search_leaders.png',out_dir)
    print('Focused refinement complete.',flush=True)


def run_diagnostic(data, out_dir):
    # Architecture, activation, loss, initialization, rate and decay are now fixed.
    # Four larger-data folds only choose training duration. The original training
    # rows are always included, so their required scaling is valid for every fold.
    selected=all_candidates(out_dir)[0]
    cfg=Config(**{**selected['config'],'hidden':tuple(selected['config']['hidden'])})
    xt,yt,xv,yv=data
    groups=np.array_split(np.random.default_rng(SEED).permutation(len(xv)),4)
    histories=[]
    for i,held in enumerate(groups):
        keep=np.concatenate([g for j,g in enumerate(groups) if j!=i])
        fold_data=(torch.cat([xt,xv[keep]]),torch.cat([yt,yv[keep]]),xv[held],yv[held])
        h,_=train_run(f'CV_fold{i+1}',cfg,fold_data,out_dir)
        histories.append(h)
    mean=pd.concat(histories).groupby('epoch',as_index=False).agg(
        val_accuracy=('val_accuracy','mean'),val_loss=('val_loss','mean'),
        fold_accuracy_std=('val_accuracy','std'))
    choice=mean.sort_values(['val_accuracy','val_loss','epoch'],ascending=[False,True,True]).iloc[0]
    mean.to_csv(out_dir/'larger_data_diagnostic.csv',index=False)
    result={'selected_run':selected['name'],'config':selected['config'],
            'selected_epoch':int(choice.epoch),'mean_heldout_accuracy':float(choice.val_accuracy),
            'mean_heldout_ce':float(choice.val_loss),
            'fold_accuracy_std':float(choice.fold_accuracy_std),
            'training_examples_per_fold':len(xt)+len(xv)-len(groups[0]),
            'heldout_examples_per_fold':len(groups[0]),
            'not_an_independent_test_estimate':True}
    (out_dir/'diagnostic_summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.5))
    for h in histories:
        axes[0].plot(h.epoch,100*h.val_accuracy,alpha=.22,color='C0')
        axes[1].plot(h.epoch,h.val_loss,alpha=.22,color='C0')
    axes[0].plot(mean.epoch,100*mean.val_accuracy,color='C0',label='Four-fold mean')
    axes[1].plot(mean.epoch,mean.val_loss,color='C0',label='Four-fold mean')
    axes[0].set_ylabel('Held-out accuracy (%)'); axes[1].set_ylabel('Held-out cross-entropy')
    for ax in axes:
        ax.axvline(int(choice.epoch),color='#555555',ls='--',lw=1,label=f'Selected epoch {int(choice.epoch)}')
        ax.set_xlabel('Epoch'); ax.grid(alpha=.22); ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out_dir/'figures'/'larger_data_diagnostic.png',dpi=190);plt.close(fig)
    print('LARGER-DATA DIAGNOSTIC:',json.dumps(result),flush=True)


def all_candidates(out_dir):
    candidates=[]
    for path in sorted((out_dir/'histories').glob('*.json')):
        meta=json.loads(path.read_text())
        if meta['name'] == 'FINAL' or meta['name'].startswith(('CV_','VALIDATION_')):
            continue  # Larger-data diagnostics and refits are not search candidates.
        c=meta['config']
        if c['loss'] != 'ce' or c['activation']=='identity' or c['initialization'] not in ('nielsen','kaiming'):
            continue
        if c['lr'] not in (.01,.03,.05) or len(c['hidden'])>3 or not set(c['hidden']) <= {8,16,32,64,128}:
            continue
        candidates.append(meta)
    return sorted(candidates,key=lambda m:(*checkpoint_key(m['best']),m['parameter_count'],m['name']))


def save_via_official_helper(model, config, data_dir, path):
    spec=importlib.util.spec_from_file_location('official_save_model_json',data_dir/'save_model_json.py')
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.save_model_json(model,hidden_layers=list(config.hidden),activation=config.activation,filename=path)


def load_json_model(path):
    payload=json.loads(path.read_text())
    model=MLP(payload['hidden_layers'],payload['activation'])
    layers=[m for m in model.modules() if isinstance(m,nn.Linear)]
    with torch.no_grad():
        for layer,w,b in zip(layers,payload['weights'],payload['biases']):
            layer.weight.copy_(torch.tensor(w,dtype=torch.float32))
            layer.bias.copy_(torch.tensor(b,dtype=torch.float32))
    return model


def final_stage(data, data_dir, out_dir):
    candidates=all_candidates(out_dir)
    if not candidates: raise RuntimeError('Run the experiments before final selection')
    selected=candidates[0]
    cfg=Config(**{**selected['config'],'hidden':tuple(selected['config']['hidden'])})
    diagnostic=json.loads((out_dir/'diagnostic_summary.json').read_text())
    assert diagnostic['selected_run']==selected['name'], 'Rerun the diagnostic after changing candidate selection'
    epoch=diagnostic['selected_epoch']
    assert 1<=epoch<=MAX_EPOCHS and cfg.loss=='ce'
    # Original-split model at the now-fixed duration; do not reselect an epoch.
    _,validation_meta=train_run('VALIDATION_SELECTED',cfg,data,out_dir,epochs=epoch,use_cache=False)
    validation_model=create_model(cfg)
    validation_model.load_state_dict(torch.load(out_dir/'checkpoints'/'VALIDATION_SELECTED_last.pt',weights_only=True))
    x_train,y_train,x_val,y_val=data
    with torch.inference_mode():
        predicted=validation_model(x_val).argmax(dim=1).numpy()
    cm=np.zeros((6,6),dtype=int)
    np.add.at(cm,(y_val.numpy(),predicted),1)
    assert cm.trace()/cm.sum() == np.mean(predicted==y_val.numpy())
    np.savetxt(out_dir/'validation_confusion.csv',cm,fmt='%d',delimiter=',')
    assert abs(metrics(validation_model,x_val,y_val,'ce')[1]-validation_meta['final']['val_accuracy'])<1e-7
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    im=ax.imshow(cm,cmap='Blues')
    for i in range(6):
        for j in range(6):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#182b3b',fontsize=10)
    ax.set(xticks=range(6),yticks=range(6),xlabel='Predicted class',ylabel='True class')
    fig.colorbar(im,ax=ax,label='Number of examples'); fig.tight_layout()
    fig.savefig(out_dir/'figures'/'validation_confusion.png',dpi=190); plt.close(fig)
    # Preprocessing was done ONCE in read_data; combine already standardized rows.
    combined_x=torch.cat([x_train,x_val]); combined_y=torch.cat([y_train,y_val])
    assert len(combined_x)==len(x_train)+len(x_val)
    _, final_meta=train_run('FINAL',cfg,(combined_x,combined_y,combined_x,combined_y),out_dir,
                           epochs=epoch,use_cache=False)
    model=create_model(cfg)
    model.load_state_dict(torch.load(out_dir/'checkpoints'/'FINAL_last.pt',weights_only=True))
    json_path=out_dir/f'{BITS_ID}.json'
    save_via_official_helper(model,cfg,data_dir,json_path)
    restored=load_json_model(json_path)
    with torch.inference_mode():
        assert torch.equal(model(combined_x),restored(combined_x))
    summary={'selected_run':selected['name'],'config':asdict(cfg),'epoch':epoch,
             'original_search_best':selected['best'],
             'validation_checkpoint':validation_meta['final'],'validation_confusion':cm.tolist(),
             'larger_data_diagnostic':diagnostic,
             'combined_training':final_meta['final'], 'candidate_count':len(candidates),
             'json_roundtrip_bitwise_equal':True,
             'post_merge_metrics_are_not_independent_validation':True,
             'ranking':[{'name':c['name'],'config':c['config'],'best':c['best'],
                         'parameters':c['parameter_count']} for c in candidates]}
    (out_dir/'final_summary.json').write_text(json.dumps(summary,indent=2))
    print('FINAL SELECTION:',json.dumps({k:v for k,v in summary.items() if k not in ('ranking','validation_confusion')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--out-dir',type=Path,default=Path('pa2_results'))
    parser.add_argument('--stage',choices=['required','search','refine','diagnostic','final','all'],default='all')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    for folder in (args.out_dir,args.out_dir/'histories',args.out_dir/'checkpoints',args.out_dir/'figures'):
        folder.mkdir(parents=True,exist_ok=True)
    data=read_data(args.data_dir,args.out_dir)
    if args.stage in ('required','all'): run_required(data,args.out_dir)
    if args.stage in ('search','all'): run_search(data,args.out_dir)
    if args.stage in ('refine','all'): run_refinement(data,args.out_dir)
    if args.stage in ('diagnostic','all'): run_diagnostic(data,args.out_dir)
    if args.stage in ('final','all'): final_stage(data,args.data_dir,args.out_dir)


if __name__=='__main__':
    main()
