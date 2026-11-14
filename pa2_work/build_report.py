"""Generate PA2 LaTeX using the same plain article layout as the PA1 report.

Run this script, then pdflatex twice on pa2_work/2024ADPS0831G.tex.
Only presentation changes; all experiment values come from existing results.
"""
from pathlib import Path
import json
from html.parser import HTMLParser
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
OUTPUT=ROOT.parent/'output'/'pdf'/'2024ADPS0831G.pdf'
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
SUMMARY=json.loads((RESULTS/'final_summary.json').read_text())
DIAG=SUMMARY['larger_data_diagnostic']
WIDTH=511.27559055118115  # Original figure/table proportions; scaled to the PA1 text block.
TEXT_WIDTH=451.27559055118115
parts=[]
pending_figure=False

def escape(text):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
             '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}',
             '^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(text))

class Markup(HTMLParser):
    def __init__(self):
        super().__init__();self.output=[]
    def handle_starttag(self,tag,attrs):
        if tag=='b':self.output.append(r'\textbf{')
        elif tag=='sub':self.output.append(r'\textsubscript{')
        elif tag=='super':self.output.append(r'\textsuperscript{')
        elif tag=='br':self.output.append(r'\newline ')
        elif tag=='link':self.output.append(r'\href{'+dict(attrs)['href']+'}{')
    def handle_endtag(self,tag):
        if tag in ('b','sub','super','link'):self.output.append('}')
    def handle_data(self,data):self.output.append(escape(data))

def latex(text):
    parser=Markup();parser.feed(str(text));return ''.join(parser.output)

def p(text,style='BodyCustom'):
    if style=='TitleCustom':
        parts.append(r'\title{Programming Assignment 2: Final Report}'+'\n'+
                     r'\author{2024ADPS0831G}'+'\n'+r'\date{}'+'\n'+r'\maketitle')
    elif text.startswith('BITS F445 - Programming Assignment 2 |'):
        pass
    else:
        parts.append(latex(text)+'\n')

def heading(text):
    titles={'A. Activation function':'Part A - Effect of Activation Function',
            'B. Cross-entropy versus quadratic cost':'Part B - Cross-Entropy versus Quadratic Cost',
            'C. Weight initialization':'Part C - Effect of Weight Initialization',
            'D. Network capacity':'Part D - Effect of Network Capacity',
            'E. L2 regularization':'Part E - Effect of L2 Regularization',
            'Selecting the final model':'Final Model Selection',
            'Checking training duration with more data':'Training Duration at a Larger Dataset Size',
            'Validation confusion matrix':'Validation Confusion Matrix',
            'Final retraining and exported model':'Final Model Training and Export',
            'Main observations':'Conclusion',
            'AI assistance and references':'Generative-AI Assistance Acknowledgement'}
    command='subsection*' if text=='Export checks' else 'section'
    parts.append('\\'+command+'{'+escape(titles.get(text,text))+'}')

def caption(text):
    global pending_figure
    if pending_figure:
        label=text.split('. ',1)[1] if text.startswith('Figure ') else text
        parts.append(r'\caption{'+latex(label)+'}\n'+r'\end{figure}')
        pending_figure=False

def newpage(): pass  # Let LaTeX flow sections and figures naturally, as in PA1.
def meta(name): return json.loads((RESULTS/'histories'/f'{name}.json').read_text())
def history(name): return pd.read_csv(RESULTS/'histories'/f'{name}.csv')
def pct(x): return f'{100*x:.2f}'
def loss(x): return f'{x:.3f}'

table_count=0
def table(rows,widths=None):
    global table_count
    captions=['Common experimental settings.',
              'Effect of hidden activation on validation performance.',
              'Cross-entropy and quadratic cost; each loss is reported on its own scale.',
              'Early and final validation performance for the initialization comparison.',
              'Network capacity comparison. All metrics are from epoch 120.',
              'Regularization results at epoch 120. The weight norm excludes biases.',
              'Leading distinct configurations, ranked by validation accuracy and then cross-entropy.',
              'Mean held-out performance in the four-fold duration diagnostic.',
              'Per-class results for the selected original-split validation model.',
              'Final model configuration.']
    count=len(rows[0]);widths=widths or [1]*count
    available=TEXT_WIDTH-8*(count-1)
    coldefs='@{}'+' '.join(r'>{\raggedright\arraybackslash}p{'+f'{available*w/sum(widths):.2f}pt'+'}' for w in widths)+'@{}'
    lines=[r'\begin{table}[htbp]',r'\centering',r'\setlength{\tabcolsep}{4pt}',
           r'\renewcommand{\arraystretch}{1.12}',r'\small' if count>=4 else r'\normalsize',
           r'\begin{tabular}{'+coldefs+'}',r'\hline']
    for i,row in enumerate(rows):
        lines.append(' & '.join(latex(cell) for cell in row)+r' \\')
        if i==0:lines.append(r'\hline')
    lines.extend([r'\hline',r'\end{tabular}',r'\caption{'+captions[table_count]+'}',r'\end{table}'])
    parts.append('\n'.join(lines));table_count+=1

def fig(filename,width=WIDTH):
    global pending_figure
    assert not pending_figure
    parts.append(r'\begin{figure}[htbp]'+'\n'+r'\centering'+'\n'+
                 r'\includegraphics[width='+f'{width/WIDTH:.4f}'+r'\linewidth]{results/figures/'+filename+'}')
    pending_figure=True


# Page 1: setup and activation comparison.
p('Training and generalization of an MLP','TitleCustom')
p('BITS F445 - Programming Assignment 2 | 2024ADPS0831G','SmallCustom')
heading('Introduction / Objective')
p('The task is six-class classification from 24 sensor measurements. The supplied split contains '
  '1,800 training and 1,800 validation examples. The experiments below isolate the requested '
  'training choices before selecting one model for retraining on all 3,600 examples.')
heading('Experimental Setup')
table([['Common setting','Value'],
       ['Optimizer / mini-batch','SGD, momentum 0 / batch size 64'],
       ['Duration / random seed','120 epochs for every controlled run / 831031'],
       ['Input scaling','Original training.csv mean and population standard deviation (ddof=0)'],
       ['Biases / output','Zero initialization / six logits, with no output activation']],
       [135,WIDTH-135])
p('Every run starts from scratch. Model initialization and mini-batch shuffling have separately '
  'reset generators using the same seed. Same-shaped models in Parts A and B have exactly matching '
  'starting parameters. Losses are measured on the complete datasets after each epoch and exclude '
  'the L2 penalty. No validation examples enter the controlled-run gradient updates.')
heading('A. Activation function')
p('Architecture 24-32-6; cross-entropy; learning rate 0.05; weight decay 0. All weights use '
  'Nielsen initialization: normal standard deviation 1/sqrt(fan-in).')
fig('A_activation.png')
caption('Figure 1. Validation accuracy and cross-entropy for the four hidden activations.')
rows=[['Activation','Best val. accuracy (%)','Best epoch','Final val. accuracy (%)','Final val. CE']]
for a,label in [('identity','Identity'),('sigmoid','Sigmoid'),('tanh','Tanh'),('relu','ReLU')]:
    m=meta(f'A_{a}');rows.append([label,pct(m['best']['val_accuracy']),m['best']['epoch'],pct(m['final']['val_accuracy']),loss(m['final']['val_loss'])])
table(rows,[80,115,65,130,WIDTH-390])
p('Identity improves quickly but levels off near 60%, consistent with its overall linear decision '
  'function. Sigmoid learns more slowly and ends close to that baseline; tanh makes a modest improvement. '
  'ReLU continues improving and reaches 75.00% at its best checkpoint. This run supports nonlinear '
  'features, particularly ReLU, for this dataset; it does not establish that every nonlinear activation '
  'will outperform a linear model under every training setup.')

# Page 2: loss comparison.
newpage();heading('B. Cross-entropy versus quadratic cost')
p('Both runs use 24-32-6, ReLU, Kaiming normal initialization, learning rate 0.05 and zero weight '
  'decay. They start with identical parameters and process the same mini-batches in the same order. '
  'Only the training objective changes.')
p('Cross-entropy receives the logits directly. The quadratic objective uses softmax probabilities '
  'and one-hot targets, with reduction:<br/><b>Q = (1 / 2n) sum over examples i and classes k of '
  '(p<sub>ik</sub> - y<sub>ik</sub>)<super>2</super>.</b><br/>'
  'The sum is over the six classes, followed by the mean over examples. There is no extra division by six.')
fig('B_loss.png')
caption('Figure 2. Shared accuracy comparison; each objective has its own validation-loss panel and scale.')
rows=[['Objective','Best val. accuracy (%)','Best epoch','Final train accuracy (%)','Final val. accuracy (%)','Final own loss']]
for name,label in [('ce','Cross-entropy'),('quadratic','Quadratic')]:
    m=meta(f'B_{name}');rows.append([label,pct(m['best']['val_accuracy']),m['best']['epoch'],pct(m['final']['train_accuracy']),pct(m['final']['val_accuracy']),loss(m['final']['val_loss'])])
table(rows,[84,95,60,100,100,WIDTH-439])
bce=history('B_ce');bq=history('B_quadratic')
p(f'At epoch 20, cross-entropy reaches {pct(bce.iloc[19].val_accuracy)}% validation accuracy, '
  f'compared with {pct(bq.iloc[19].val_accuracy)}% for quadratic cost. At epoch 120 the gap remains '
  f'{100*(bce.iloc[-1].val_accuracy-bq.iloc[-1].val_accuracy):.2f} percentage points. Quadratic cost '
  'is still improving toward the end of its run, but it has not caught up within the required budget.')
p('This difference is consistent with the objectives having different gradients. For softmax '
  'cross-entropy the logit gradient is proportional to p-y. Quadratic cost differentiates through '
  'the softmax Jacobian as well, which can weaken updates for confident mistakes. With the learning '
  'rate held fixed, the observed learning speed favors cross-entropy. A separate learning-rate '
  'search for quadratic cost was not part of this controlled comparison.')
p('The numerical loss values in the last column have different meanings and scales. The quadratic '
  'value is not evidence that it is the better model; accuracy and the progress within each curve '
  'provide the appropriate comparisons here.')

# Page 3: initialization, including the early epochs.
newpage();heading('C. Weight initialization')
p('Architecture 24-64-32-6; ReLU; cross-entropy; learning rate 0.03; weight decay 0. '
  'All layers, including the output layer, use the specified initialization. The normal standard '
  'deviations are 0.01, 1, and 1/sqrt(fan-in); Kaiming uses fan-in mode for ReLU.')
fig('C_initialization.png')
caption('Figure 3. Complete validation trajectories. Large-normal initialization starts with much higher loss.')
fig('C_early.png')
caption('Figure 4. The same runs restricted to epochs 1-20 to show the initial learning behavior.')
rows=[['Initialization','Epoch 1 val. CE','Epoch 5 val. acc. (%)','Best val. acc. (%)','Final val. acc. (%)']]
for init,label in [('small','Small normal'),('large','Large normal'),('nielsen','Nielsen'),('kaiming','Kaiming')]:
    m=meta(f'C_{init}');h=history(f'C_{init}')
    rows.append([label,loss(h.iloc[0].val_loss),pct(h.iloc[4].val_accuracy),pct(m['best']['val_accuracy']),pct(m['final']['val_accuracy'])])
table(rows,[100,90,110,105,WIDTH-405])
p('Small-normal weights make very little early progress: validation loss stays close to ln(6) '
  'and final accuracy is only 28.72%. Large-normal weights start with validation CE 20.630; '
  'the run stays finite but ends at CE 2.306 and 51.56% accuracy. Fan-in scaling avoids these '
  'extremes. Kaiming learns faster at first, while Nielsen finishes better in this fixed-seed run '
  '(73.67% versus 71.50%). Thus the fastest start did not give the best final validation result.')

# Page 4: capacity and overfitting.
newpage();heading('D. Network capacity')
p('ReLU, cross-entropy, Kaiming normal initialization, learning rate 0.03 and zero weight decay '
  'are fixed. Increasing capacity greatly reduces training error, but the validation improvement '
  'is much smaller.')
rows=[['Model / hidden layers','Parameters','Train acc. (%)','Val. acc. (%)','Train CE','Val. CE']]
for name,label in [('tiny','Tiny / 8'),('small','Small / 32'),('medium','Medium / 64-32'),('large','Large / 128-128')]:
    m=meta(f'D_{name}');f=m['final']
    rows.append([label,f'{m["parameter_count"]:,}',pct(f['train_accuracy']),pct(f['val_accuracy']),loss(f['train_loss']),loss(f['val_loss'])])
table(rows,[125,70,90,85,70,WIDTH-440])
caption('Final capacity comparison. All metrics are from epoch 120, rather than each model\'s best checkpoint.')
fig('D_capacity.png')
caption('Figure 5. Validation accuracy and cross-entropy as network size increases.')
fig('D_large_curves.png')
caption('Figure 6. Training and validation curves for the unregularized Large model.')
p('Tiny has both low training accuracy and low validation accuracy, suggesting insufficient '
  'capacity within this budget. Small improves both. Medium and Large nearly fit the training '
  'set, yet finish with validation accuracy only slightly above Small and with worse validation '
  'cross-entropy. Large reaches 99.94% training accuracy against 72.89% validation accuracy. '
  'Its training CE continues down while validation CE turns upward: capacity alone is not enough. '
  'The growing loss gap also shows why accuracy by itself can conceal worsening confidence on errors.')

# Page 5: decay.
newpage();heading('E. L2 regularization')
p('The model is 24-128-128-6 with ReLU, cross-entropy, Kaiming normal initialization and '
  'learning rate 0.03. The SGD weight-decay value applies to <b>all trainable parameters, including '
  'biases</b>, as required. Every run uses the same initial parameters and batch sequence.')
fig('E_decay.png')
caption('Figure 7. Validation accuracy and cross-entropy for the five required weight-decay values.')
rows=[['Weight decay','Final train acc. (%)','Final val. acc. (%)','Final val. CE','Weight L2 norm']]
for wd in (0.,.001,.01,.03,.1):
    f=meta(f'E_wd{wd:g}')['final']
    rows.append([f'{wd:g}',pct(f['train_accuracy']),pct(f['val_accuracy']),loss(f['val_loss']),f'{f["weight_l2"]:.3f}'])
table(rows,[85,120,120,90,WIDTH-415])
p('The reported weight norm is sqrt(sum over all weight-matrix entries of w<super>2</super>), '
  'excluding biases. It is one norm of the concatenated weight matrices, rather than the sum of '
  'the individual layer norms. Biases are still regularized by the optimizer.')
p('With no decay or 0.001, the model fits almost all training examples while validation CE '
  'rises late in training. Decay 0.01 reduces the final weight norm from 25.485 to 12.092, improves '
  'validation accuracy from 72.89% to 76.06%, and reduces validation CE from 0.922 to 0.613. '
  'This is the most useful balance among the five prescribed values.')
p('Decay 0.03 reduces the gap further but also reduces training accuracy to 81.94%; its final '
  'validation accuracy is 73.11%. At 0.1 the weight norm falls to 1.785 and both datasets perform '
  'poorly. Its early validation peak does not persist: by epoch 120 it is clearly underfitting. '
  'The experiment supports moderate regularization for this model, rather than either no '
  'regularization or the strongest available value.')

# Page 6: declared additional search.
newpage();heading('Selecting the final model')
p('Parts A-E point to ReLU, cross-entropy, fan-in-scaled initialization and moderate weight '
  'decay. The extension first compared ReLU/Kaiming and tanh/Nielsen at hidden widths 32, '
  '64-32, 128-128 and 128-64-32, with learning rate 0.03 and decay 0.001 or 0.01 '
  '(16 additional runs). Tanh did not match the leading ReLU models.')
p('A focused follow-up used the two leading distinct architectures (128-128 and 64-32). '
  'It checked learning rates 0.01 and 0.05 with decay 0.01 or 0.03, plus rate 0.03 with '
  'decay 0.005 or 0.02. Six further checks used ReLU/Nielsen with widths 64, 128 or 128-128 '
  'and decay 0.01 or 0.03 at rate 0.03. These 18 follow-up runs used the same seed, batch size '
  'and 120-epoch budget; no prohibited optimizer or architecture was introduced.')
unique=[];seen=set()
for m in SUMMARY['ranking']:
    key=json.dumps(m['config'],sort_keys=True)
    if key not in seen:unique.append(m);seen.add(key)
rows=[['Hidden layers','Init.','LR','Decay','Best epoch','Val. acc. (%)','Val. CE']]
for m in unique[:7]:
    c=m['config'];b=m['best']
    rows.append(['-'.join(map(str,c['hidden'])),c['initialization'].capitalize(),f'{c["lr"]:g}',
                 f'{c["weight_decay"]:g}',b['epoch'],pct(b['val_accuracy']),loss(b['val_loss'])])
table(rows,[80,75,45,50,70,100,WIDTH-420])
caption('Leading distinct configurations, ranked by best validation accuracy, then cross-entropy. All use ReLU.')
fig('search_leaders.png')
caption('Figure 8. Full validation curves for the leading configurations; isolated accuracy peaks are small relative to the overall trends.')
p('The two leading configurations tie at 77.39% accuracy. The Nielsen model has lower '
  'cross-entropy (0.596 versus 0.669), so the selected architecture is <b>24-128-128-6</b> with '
  '<b>ReLU, Nielsen initialization, learning rate 0.03 and weight decay 0.01</b>. A smaller '
  '64-32 model reaches 77.00%, so the advantage of the larger model is modest. The six-digit '
  'seed remains fixed; no seed search, ensemble, parameter averaging or learning-rate schedule '
  'was used. Validation has been used repeatedly for selection, so these scores should not '
  'be treated as unbiased estimates of hidden-test performance.')

# Page 7: bigger-data diagnostic.
newpage();heading('Checking training duration with more data')
p('After fixing the architecture and all other hyperparameters, training duration was checked '
  'using four larger-data folds. The 1,800 original training examples remain in every fold. '
  'The validation rows are shuffled with seed 831031 into four groups of 450. Each fold trains '
  'on the original training rows plus 1,350 validation rows and holds out the remaining 450 '
  'rows: 3,150 training examples per fold. No held-out labels are used in its gradient updates.')
p('All four models start from scratch and train for 120 epochs. Input scaling stays fixed to '
  'the original training.csv statistics, because the original training rows are present in every '
  'fold. The mean held-out accuracy across folds selects the duration; ties use mean held-out '
  'cross-entropy, then the earlier epoch. No setting other than duration is adjusted here.')
fig('larger_data_diagnostic.png')
caption('Figure 9. Faint lines show individual folds; solid lines show the four-fold mean. The dashed line marks the selected epoch.')
dh=pd.read_csv(RESULTS/'larger_data_diagnostic.csv')
epochs=sorted(set([30,60,90,106,int(SUMMARY['epoch']),120]))
rows=[['Epoch','Mean held-out accuracy (%)','Mean held-out CE']]
for epoch in epochs:
    r=dh[dh.epoch==epoch].iloc[0]
    rows.append([epoch,pct(r.val_accuracy),loss(r.val_loss)])
table(rows,[75,245,WIDTH-320])
p(f'The selected duration is <b>{SUMMARY["epoch"]} epochs</b>, at mean held-out accuracy '
  f'<b>{pct(DIAG["mean_heldout_accuracy"])}%</b> and CE {loss(DIAG["mean_heldout_ce"])}. '
  f'The sample standard deviation of accuracy across the four folds is '
  f'{100*DIAG["fold_accuracy_std"]:.2f} percentage points. The original 1,800-row search had '
  'its best checkpoint at epoch 106; the larger-data diagnostic supports a nearby duration '
  'rather than an unrestricted extension beyond the assignment budget.')
p('This is a model-selection diagnostic, not a new independent test. Architecture and '
  'hyperparameters were already chosen using the supplied validation set. The folds also '
  'share many training rows. Their mean is useful for choosing a duration at a training size '
  'closer to 3,600, but it does not remove the optimism caused by earlier validation-based selection.')

# Page 8: confusion matrix for the original split, at the finally selected duration.
newpage();heading('Validation confusion matrix')
v=SUMMARY['validation_checkpoint'];cm=np.asarray(SUMMARY['validation_confusion'])
p(f'To evaluate the fully selected configuration, a fresh model was trained on the original '
  f'1,800 training rows for exactly {SUMMARY["epoch"]} epochs. This duration was fixed by the '
  f'larger-data diagnostic; no new checkpoint selection was made from this run. Its training '
  f'accuracy is {pct(v["train_accuracy"])}% and validation accuracy is '
  f'<b>{pct(v["val_accuracy"])}%</b> ({cm.trace()} of {cm.sum()} examples). '
  f'Training and validation CE are {loss(v["train_loss"])} and {loss(v["val_loss"])}.')
fig('validation_confusion.png',width=365)
caption('Figure 10. Confusion matrix on the original validation.csv. Rows are true labels and columns are predictions.')
recall=cm.diagonal()/cm.sum(1);precision=cm.diagonal()/cm.sum(0)
rows=[['Class','Operating condition','Support','Correct','Recall (%)','Precision (%)']]
for i in range(6):
    rows.append([i,'Normal' if i==0 else f'Condition {chr(64+i)}',cm[i].sum(),cm[i,i],pct(recall[i]),pct(precision[i])])
table(rows,[45,160,65,65,85,WIDTH-420])
best=int(recall.argmax());worst=int(recall.argmin());off=cm.copy();np.fill_diagonal(off,0)
i,j=np.unravel_index(off.argmax(),off.shape)
p(f'Class {best} is the most reliably recovered by recall ({pct(recall[best])}%). '
  f'Class {worst} has the lowest recall ({pct(recall[worst])}%). The largest directional '
  f'confusion is true class {i} predicted as {j}: {cm[i,j]} examples '
  f'({100*cm[i,j]/cm[i].sum():.2f}% of class {i}). These counts show where the remaining '
  'errors concentrate; they do not establish a physical explanation because the sensor features '
  'and operating conditions are anonymized.')
p('This 111-epoch original-split score differs from the 77.39% search peak at epoch 106. '
  'Both are reported explicitly: the architecture comes from the search, while the final '
  'duration comes from the larger-data diagnostic. The confusion matrix is from the original '
  'split before merging, not from a model that has already trained on validation.csv.')

# Page 9: final model and verification, acknowledgement, references.
newpage();heading('Final retraining and exported model')
c=SUMMARY['config'];f=SUMMARY['combined_training']
table([['Final setting','Selected value'],['Architecture','24-128-128-6 (20,486 trainable parameters)'],
       ['Hidden activation / output','ReLU in both hidden layers / six logits'],
       ['Initialization / biases','Nielsen normal, standard deviation 1/sqrt(fan-in) / zero biases'],
       ['Loss / optimizer','Cross-entropy / SGD without momentum'],
       ['Learning rate / mini-batch','0.03 / 64'],['Weight decay','0.01 on all weights and biases'],
       ['Epochs / seed',f'{SUMMARY["epoch"]} / 831031'],
       ['Training data','All 3,600 supplied examples'],
       ['Normalization','Original training.csv statistics only; no recomputation after merging']],
       [145,WIDTH-145])
p('All model choices were fixed before final retraining. A new model was initialized from '
  'scratch and trained on the combined, already-standardized rows for the selected number '
  'of epochs. The final weights are from the last epoch of that run. No combined-data metric '
  'was used for another round of model or epoch selection.')
p(f'At the final epoch, combined-data training accuracy is <b>{pct(f["train_accuracy"])}%</b> '
  f'and cross-entropy is <b>{loss(f["train_loss"])}</b>. These are training diagnostics. '
  'A checker score on the original validation rows is also an in-sample sanity check now '
  'that those rows were included in final training; it is not a new generalization estimate.')
heading('Export checks')
p('The provided save_model_json.py helper writes the architecture, activation, weight '
  'matrices and biases in x1 through x24 feature order. Reloading the JSON gives bitwise '
  'identical PyTorch logits on all 3,600 standardized examples. The supplied '
  'check_model_json.py accepts the exported file. Additional checks confirm zero initial '
  'biases, identical starting parameters for Parts A and B, identical histories for repeated '
  'controlled configurations, and unchanged training-only preprocessing. All losses and '
  'parameters are finite. No PA1 test data were used in any PA2 experiment.')
heading('Main observations')
p('Nonlinearity is useful on this task, but the activation and initialization affect how '
  'quickly the model learns within a fixed budget. Cross-entropy gives better accuracy than '
  'quadratic cost under the prescribed comparison. Larger models fit the training data '
  'much more closely, so their additional capacity needs regularization. Moderate decay '
  'improves the validation result; excessive decay removes useful capacity. The final choice '
  'combines this evidence with a held-out duration check at a larger training-set size. '
  'The hidden 50,000-example test set was unavailable, so no hidden-test accuracy or mark '
  'is claimed.')
heading('AI assistance and references')
p('OpenAI Codex assisted with implementation, experiment execution, analysis, figures and '
  'report drafting. The numerical results and plots were produced by running the program '
  'on the supplied CSV files; the export was verified with the supplied checker.','SmallCustom')
heading('References')
p('1. BITS F445, Programming Assignment 2 question paper and supporting scripts.<br/>'
  '2. PyTorch 2.14 documentation: '
  '<link href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html">CrossEntropyLoss</link>, '
  '<link href="https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html">SGD</link>, and '
  '<link href="https://docs.pytorch.org/docs/2.14/nn.init.html">weight initialization</link>.<br/>'
  '3. Michael Nielsen, Neural Networks and Deep Learning, Chapter 3 (quadratic cost convention).',
  'SmallCustom')


assert not pending_figure and table_count==10
preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{array}
\usepackage[section]{placeins}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Programming Assignment 2: Final Report},pdfauthor={2024ADPS0831G}}
\pagestyle{plain}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{5}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.7}
\begin{document}
'''
tex_path=ROOT/'2024ADPS0831G.tex'
tex_path.write_text(preamble+'\n\n'.join(parts)+'\n'+r'\end{document}'+'\n',encoding='utf-8')
print(tex_path)
