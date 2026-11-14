# BITS F445 PA2: training and generalization of an MLP

Course project for six-class classification from 24 anonymized sensor features.
Includes the implementation, report, trained model and recorded experiment evidence.

**[Download the full comparison ZIP](downloads/PA2_2024ADPS0831G_full_comparison.zip)**

## Deliverables

- [Report: 2024ADPS0831G.pdf](output/pa2_submission/2024ADPS0831G.pdf)
- [Python program](output/pa2_submission/2024ADPS0831G.py)
- [Final trained JSON](output/pa2_submission/2024ADPS0831G.json)

The report uses a plain LaTeX article layout. The three deliverables are kept
together in `output/pa2_submission/`. The comparison ZIP contains supporting
evidence as well; it is not the three-file submission itself.

## Experiments

The required Parts A-E compare activations, cross-entropy and quadratic cost,
initialization, model capacity, and L2 regularization. Additional comparisons stay
within the assignment's permitted model and training choices. Models use SGD
without momentum, batch size 64, zero initial biases, seed 831031, and at most
120 epochs. The input mean and population standard deviation are always those
computed from the original 1,800 training rows.

After hyperparameter selection, a four-fold larger-data diagnostic chooses the
training duration. It partitions the original 1,800 validation rows into groups
of 450. Each fold trains on all original training rows plus the other 1,350
validation rows (3,150 total), holding out 450 rows. This is a duration-selection
diagnostic, not ordinary four-fold cross-validation over all 3,600 rows, and not
an independent test estimate after earlier validation-based model selection.

## Selected model and measured results

Architecture: **24 -> 128 -> 128 -> 6**, ReLU, Nielsen normal initialization,
cross-entropy, learning rate **0.03**, weight decay **0.01** on all parameters,
batch size **64**, seed **831031**, final duration **111 epochs**.

| Measurement | Accuracy | Meaning |
|---|---:|---|
| Best original-split search checkpoint | 77.39% | Validation score at epoch 106 |
| Original-split training at the fixed 111 epochs | 94.67% | Training score of the confusion-matrix model |
| Original-split validation at the fixed 111 epochs | 76.00% | Validation score of the confusion-matrix model |
| Four-fold larger-data diagnostic | 80.83% | Mean held-out accuracy at epoch 111, used for duration selection |
| Final combined-data training | 91.86% | One fresh model trained on all 3,600 rows |
| Final JSON checker on original validation rows | 91.11% | In-sample sanity check after combined retraining |

The hidden PA2 test set was unavailable. No hidden-test accuracy or mark is claimed.

## Reproduction

From the repository root, using a Python environment with the packages listed in
`requirements.txt` already available:

```bash
python pa2_work/2024ADPS0831G.py --data-dir inputs/PA2_supporting_files --out-dir pa2_work/reproduction --stage all
```

The stages are `required`, `search`, `refine`, `diagnostic`, and `final`, in that
order. A fresh output directory avoids reuse of cached histories. The cache only
checks configuration and epoch count; changing data or code warrants a fresh
directory. The original run used Python 3.12.3 and PyTorch 2.14.0+cu130, running
on CPU in WSL with one thread. Floating-point trajectories may vary across
versions or hardware.

To check the packaged final model with the course-provided checker:

```bash
cd inputs/PA2_supporting_files
python check_model_json.py ../../output/pa2_submission/2024ADPS0831G.json
```

The checker passed. An independent NumPy evaluation and the specification audit
are recorded in `pa2_work/results/verification.json`.

## Repository contents

- `pa2_work/results/histories/`: per-epoch CSV metrics and JSON configurations.
- `pa2_work/results/checkpoints/`: selected and final model state dictionaries.
- `pa2_work/results/figures/`: report figures.
- `pa2_work/results/final_summary.json`: complete selection results.
- `pa2_work/results/data_audit.json`: dataset hashes, normalization and environment.
- `inputs/PA2_supporting_files/`: course-provided CSVs, export/check helpers and example.
- `inputs/PA2_question.pdf`: supplied assignment brief.
- `pa2_work/build_report.py` and `pa2_work/2024ADPS0831G.tex`: report source.
- `MANIFEST.json`: SHA-256 hashes of the original comparison-package files.
- `downloads/`: the full comparison ZIP.

To regenerate the LaTeX source, run `python pa2_work/build_report.py`. Compile
`2024ADPS0831G.tex` twice with `pdflatex` from `pa2_work/`. To audit the recorded
results, run `python pa2_work/verify_results.py` from the repository root.

Course-provided data and helper scripts retain their original attribution.
OpenAI Codex assisted with implementation, experiment execution, analysis,
figures and report drafting; the report acknowledges that assistance.
