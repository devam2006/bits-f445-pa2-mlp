"""Package the PA2 implementation and evidence, without Python environments."""
from pathlib import Path
import hashlib
import json
import zipfile

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'output' / 'PA2_2024ADPS0831G_full_comparison.zip'
files = {}

for name in ('2024ADPS0831G.py', 'build_report.py', 'verify_results.py',
             'create_comparison_bundle.py', '2024ADPS0831G.tex'):
    path = BASE / 'pa2_work' / name
    files[path.relative_to(BASE).as_posix()] = path
for folder in ('pa2_work/results', 'inputs/PA2_supporting_files', 'output/pa2_submission'):
    for path in sorted((BASE / folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            files[path.relative_to(BASE).as_posix()] = path
question = Path.home() / 'Downloads' / 'PA2_question.pdf'
if question.is_file():
    files['inputs/PA2_question.pdf'] = question

readme = '''# PA2 full comparison package - 2024ADPS0831G

This archive is for reviewing and reproducing the project. The three deliverables
are in output/pa2_submission/. Upload those three files individually if submitting;
this larger comparison archive includes extra evidence and supporting files.

Contents:
- output/pa2_submission/: final PDF, Python program and trained JSON model.
- pa2_work/2024ADPS0831G.py: experiment, selection, retraining and export program.
- pa2_work/results/histories/: per-epoch CSVs and configuration/metric JSONs.
- pa2_work/results/checkpoints/: best and final model parameters for the runs.
- pa2_work/results/figures/: all plots used in the report.
- pa2_work/results/final_summary.json: complete model selection and final metrics.
- pa2_work/results/larger_data_diagnostic.csv: four-fold duration results.
- pa2_work/results/verification.json: recorded technical verification results.
- pa2_work/results/data_audit.json and preprocessing.npz: original data hashes,
  environment details and training-only normalization statistics.
- pa2_work/build_report.py and 2024ADPS0831G.tex: report generation and LaTeX source.
- pa2_work/verify_results.py: independent NumPy evaluation/specification audit.
- inputs/PA2_supporting_files/: the supplied CSVs, official save/check helpers,
  example JSON and README. The assignment question paper is included if available.
- MANIFEST.json: SHA-256 hashes and sizes for the archived source and evidence.

How to interpret the numbers:
- Best original-split search checkpoint: 77.39% validation at epoch 106.
- Original-split refit at selected final duration (111 epochs):
  94.67% training and 76.00% validation accuracy; this provides the confusion matrix.
- Optional larger-data diagnostic: mean 80.83% held-out accuracy at epoch 111.
  Four groups of 450 validation rows are used in turn as held-out data. Each fold
  trains on all 1,800 original training rows plus the other 1,350 validation rows.
  This chooses duration after hyperparameter selection; it is not an independent
  test estimate or ordinary four-fold CV over all 3,600 rows.
- Final fresh model trained on all 3,600 rows: 91.86% training accuracy.
- Final JSON checker: 91.11% on the original validation rows. Those rows were
  included in final training, so this is only an in-sample sanity check.
- No hidden PA2 test set was available.

Selected model:
24 -> 128 -> 128 -> 6; ReLU; Nielsen initialization; zero initial biases;
cross-entropy; SGD without momentum; learning rate 0.03; batch size 64;
weight decay 0.01 on all trainable parameters; seed 831031; 111 epochs.

Reproduce from the extracted archive root using an existing Python environment
with torch, numpy, pandas and matplotlib:

    python pa2_work/2024ADPS0831G.py --data-dir inputs/PA2_supporting_files --out-dir pa2_work/reproduction --stage all

Use a new output directory for an independent rerun. Cached runs match the
configuration and epoch count; they do not check for edited data/code. The
original run used Python 3.12.3 and PyTorch 2.14.0+cu130 on CPU in WSL, with one
thread. Floating-point results may vary across versions or hardware. No Python
environment or installed library files are included in this archive.

Check the packaged final model independently from inputs/PA2_supporting_files/:

    python check_model_json.py ../../output/pa2_submission/2024ADPS0831G.json

To audit the recorded results, run python pa2_work/verify_results.py from the root.
To rebuild the report with existing results, run python pa2_work/build_report.py
(requires numpy and pandas), then run pdflatex twice on 2024ADPS0831G.tex from
pa2_work/. The provided PDF uses the simple LaTeX article style requested.

The report accurately acknowledges AI assistance. No files are transmitted to
anyone by creating this archive; you can share it yourself for comparison.
'''

manifest = {name: {'bytes': path.stat().st_size,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sorted(files.items())}
with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('README.md', readme)
    for name, path in sorted(files.items()):
        archive.write(path, arcname=name)
    archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2))

with zipfile.ZipFile(OUT) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(files) + 2
    for name, entry in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256']
    submission = [name for name in archive.namelist() if name.startswith('output/pa2_submission/')]
    assert len(submission) == 3

print(f'Created: {OUT}')
print(f'Files: {len(files) + 2}; ZIP size: {OUT.stat().st_size / 1024**2:.2f} MiB')
print('Archive integrity, all file hashes and three deliverables verified.')
