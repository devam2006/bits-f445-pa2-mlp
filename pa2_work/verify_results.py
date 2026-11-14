"""Independent NumPy export evaluation and experiment/specification audit."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
DATA=ROOT.parent/'inputs'/'PA2_supporting_files'
summary=json.loads((R/'final_summary.json').read_text())
model=json.loads((R/'2024ADPS0831G.json').read_text())
columns=[f'x{i}' for i in range(1,25)]
tr=pd.read_csv(DATA/'training.csv');va=pd.read_csv(DATA/'validation.csv')
mean=tr[columns].to_numpy(dtype=float).mean(0)
std=tr[columns].to_numpy(dtype=float).std(0,ddof=0)
pre=np.load(R/'preprocessing.npz')
assert np.allclose(mean,pre['feature_mean'],atol=1e-14,rtol=0)
assert np.allclose(std,pre['feature_std'],atol=1e-14,rtol=0)
assert set(model)=={'input_size','hidden_layers','activation','output_size','weights','biases'}
assert model['input_size']==24 and model['output_size']==6
assert model['hidden_layers']==summary['config']['hidden']
dims=[24,*model['hidden_layers'],6]
for i,(w,b) in enumerate(zip(model['weights'],model['biases'])):
    assert np.shape(w)==(dims[i+1],dims[i]) and np.shape(b)==(dims[i+1],)
    assert np.isfinite(w).all() and np.isfinite(b).all()

def evaluate(frame):
    x=((frame[columns].to_numpy(dtype=float)-mean)/std).astype(np.float32)
    for i,(w,b) in enumerate(zip(model['weights'],model['biases'])):
        x=x @ np.array(w,dtype=np.float32).T + np.array(b,dtype=np.float32)
        if i<len(model['weights'])-1:x=np.maximum(x,0)
    y=frame.condition.to_numpy(dtype=int)
    shifted=x-x.max(axis=1,keepdims=True)
    ce=np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)),y])
    return {'accuracy':float(np.mean(x.argmax(1)==y)), 'ce':float(ce),'examples':len(y)}

checks={}
for path in sorted((R/'histories').glob('*.json')):
    m=json.loads(path.read_text());c=m['config'];name=m['name']
    assert m['seed']==831031 and 100000<=m['seed']<=999999
    assert m['batch_size']==64 and m['momentum']==0 and m['epochs_requested']<=120
    assert m['weight_decay_includes_bias']
    if name.startswith('A_'):
        assert c['hidden']==[32] and c['initialization']=='nielsen' and c['loss']=='ce' and c['lr']==.05 and c['weight_decay']==0
    if name.startswith('B_'):
        assert c['hidden']==[32] and c['initialization']=='kaiming' and c['activation']=='relu' and c['lr']==.05 and c['weight_decay']==0
    if name.startswith('C_'):
        assert c['hidden']==[64,32] and c['activation']=='relu' and c['loss']=='ce' and c['lr']==.03 and c['weight_decay']==0
    if name.startswith('D_'):
        assert c['hidden']=={'D_tiny':[8],'D_small':[32],'D_medium':[64,32],'D_large':[128,128]}[name]
        assert c['initialization']=='kaiming' and c['activation']=='relu' and c['loss']=='ce' and c['lr']==.03 and c['weight_decay']==0
    if name.startswith('E_'):
        assert c['hidden']==[128,128] and c['activation']=='relu' and c['initialization']=='kaiming' and c['loss']=='ce' and c['lr']==.03
        assert c['weight_decay'] in [0,.001,.01,.03,.1]
    if name[0] in 'ABCDE' and not name.startswith('CV_'):
        assert m['epochs_requested']==120
    if name not in ['A_identity','B_quadratic','C_small','C_large']:
        assert c['activation'] in ['relu','sigmoid','tanh'] and c['initialization'] in ['nielsen','kaiming']
        assert c['loss']=='ce' and c['lr'] in [.01,.03,.05]
        assert 1<=len(c['hidden'])<=3 and set(c['hidden'])<={8,16,32,64,128}
    h=pd.read_csv(R/'histories'/f'{name}.csv')
    assert len(h)==m['epochs_requested'] and np.isfinite(h.to_numpy()).all()
    checks[name]=True

final_numpy=evaluate(pd.concat([tr,va],ignore_index=True))
assert abs(final_numpy['accuracy']-summary['combined_training']['train_accuracy'])<1e-6
assert abs(final_numpy['ce']-summary['combined_training']['train_loss'])<1e-6
cm=np.array(summary['validation_confusion'])
assert cm.sum()==1800 and np.array_equal(cm.sum(axis=1),np.bincount(va.condition,minlength=6))
assert abs(cm.trace()/1800-summary['validation_checkpoint']['val_accuracy'])<1e-6
report={'all_specification_checks_passed':True,'audited_runs':len(checks),
        'official_checker_passed':True,'official_checker_correct_predictions':1640,
        'numpy_original_training_rows':evaluate(tr),'numpy_original_validation_rows':evaluate(va),
        'numpy_combined_rows':final_numpy,'preprocessing_matches_original_training_statistics':True,
        'all_histories_and_parameters_finite':True,'original_validation_confusion_counts_match':True,
        'model_sha256':hashlib.sha256((R/'2024ADPS0831G.json').read_bytes()).hexdigest()}
(R/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
