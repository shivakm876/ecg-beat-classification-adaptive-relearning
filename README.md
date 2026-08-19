# ECG Beat Classification with Adaptive Re-learning

## 1. Project Overview

This project implements a five-class ECG beat classification pipeline using the MIT-BIH Arrhythmia Database. ECG beats are preprocessed, extracted around annotation locations, mapped to the AAMI classes **N, S, V, F, and Q**, and classified using a PyTorch 1D CNN.

The project also implements an adaptive re-learning workflow with SQLite-based prediction persistence, simulated reviewer corrections, an append-only audit trail, a rule-based re-learning trigger, locked DS2 evaluation, catastrophic-forgetting analysis, model versioning, and a model promotion gate.

---

## 2. Architecture & Design Decisions

### Overall flow

```text
MIT-BIH Arrhythmia Database
            |
            v
     ECG Preprocessing
        0.5–40 Hz
            |
            v
      Beat Extraction
  90 samples before +
  180 samples after
            |
            v
       AAMI Mapping
      N / S / V / F / Q
            |
            v
   Record-level DS1 / DS2
         /       \
        v         v
      DS1      LOCKED DS2
        |           |
        v           |
Train / Validation  |
        |           |
        v           |
     cnn_v1.0       |
        |           |
        v           |
 DS1 Adaptation     |
      Pool           |
        |             |
        v             |
Reviewer Corrections |
        |             |
        v             |
   Audit Trail        |
        |             |
        v             |
  Trigger Policy      |
        |             |
        v             |
   Re-learning        |
        |             |
   +----+----+        |
   |         |        |
Reviewer   DS1 Replay |
Samples             |
   |         |        |
   +----+----+        |
        |             |
        v             |
     cnn_v2.0 --------+
                      |
                      v
             Locked DS2 Evaluation
                      |
                      v
            Forgetting Analysis
                      |
                      v
                Promotion Gate
                  /       \
             Reject       Promote
                            |
                            v
                      Active Model
                            |
                            v
                  New External Beat
                       Inference

Key design decisions
AAMI class mapping

MIT-BIH annotation symbols are mapped to five AAMI-style classes:

N — Normal
S — Supraventricular
V — Ventricular
F — Fusion
Q — Unknown / other

The assignment specifically recommends the AAMI five-class mapping.

ECG preprocessing

The continuous ECG signal is processed using a third-order Butterworth bandpass filter:

High-pass: 0.5 Hz
Low-pass: 40 Hz

Each beat is extracted around the annotated beat location using:

90 samples before the annotation
180 samples after the annotation
-------------------------------
270 samples per beat
Record-level dataset separation

The implementation separates records into DS1 and DS2 rather than randomly splitting individual beats.

Within DS1, the notebook further creates:

17 records → training
5 records  → validation

The implementation checks for record overlap between training, validation, and DS2.

Baseline model

The classifier is a PyTorch 1D CNN designed for fixed-length ECG beat morphology.

The architecture contains four convolutional layers with batch normalization, ReLU activations, max pooling and dropout, followed by adaptive average pooling and fully connected layers:

Conv1D: 1 → 32, kernel 7
Conv1D: 32 → 64, kernel 5
Conv1D: 64 → 128, kernel 3
Conv1D: 128 → 128, kernel 3


AdaptiveAvgPool1d
Linear: 128 → 64
Linear: 64 → 5

Training uses Adam and weighted cross-entropy loss.

Class imbalance

The DS1 training data is imbalanced. The implementation:

oversamples S and F in the DS1 training data;
does not oversample validation;
does not modify DS2;
calculates moderated square-root inverse-frequency class weights;
uses the resulting weights in the final training loss.
Reviewer correction workflow

Reviewer corrections are simulated within the notebook using known reference annotations for the DS1 adaptation pool.

The corrected labels are recorded in an append-only SQLite audit trail rather than silently overwriting history.

Re-learning trigger

The re-learning workflow uses explicit correction-based rules rather than retraining after every individual correction.

The configured trigger considers:

5 unique corrections
2 minority-class corrections
1% correction drift
Re-learning strategy

When re-learning is triggered, the new model starts from cnn_v1.0 and uses:

Reviewer-corrected adaptation samples
+
Class-aware DS1 replay

The locked DS2 evaluation set is excluded from re-learning.

Catastrophic forgetting and promotion

After re-learning, cnn_v1.0 and cnn_v2.0 are evaluated on the same locked DS2 dataset.

The promotion gate checks:

Accuracy
Macro F1
Per-class F1

The configured maximum regression tolerance is:

1 percentage point

F and Q are evaluated normally. Their zero F1 values are reported as a class-performance/data-sparsity limitation rather than excluded.

3. Repository / Notebook Structure

The current implementation is provided as a Google Colab/Jupyter notebook:

ecg-beat-classification-adaptive-relearning/
└── ECG_final.ipynb

The notebook contains the complete Part A, Part B, and Part C workflow.

Logical notebook structure
ECG_final.ipynb


├── Dataset preparation
│   ├── MIT-BIH data access
│   ├── AAMI mapping
│   ├── ECG preprocessing
│   └── Beat extraction
│
├── Part A
│   ├── DS1 / DS2 separation
│   ├── DS1 train / validation split
│   ├── Class imbalance handling
│   ├── 1D CNN training
│   └── DS2 evaluation
│
├── Part B
│   ├── Baseline model saving
│   ├── SQLite initialization
│   ├── Baseline inference
│   └── Prediction persistence
│
└── Part C
    ├── C0 — classified beat inspection
    ├── C1 — append-only audit trail
    ├── C2 — reviewer reclassification
    ├── C3 — re-learning trigger
    ├── C4 — locked DS2 + adaptation data
    ├── C5 — adaptive re-learning
    ├── C6 — before/after evaluation
    ├── C6.5 — catastrophic forgetting analysis
    ├── C7 — validation / promotion
    └── C8 — active-model inference
Main generated artifacts
cnn_v1.0_baseline.pt
cnn_v2.0_relearned.pt
ecg_pipeline.db

The SQLite database contains persistence for classified beats, reclassification audit records, model registry information, and promotion events.

The classified_beats records include the unique beat ID, source record, timestamp, input signal reference, predicted label, confidence, and model version.

4. Environment Setup

The notebook was developed and executed in Google Colab and uses Google Drive for dataset and generated-artifact storage.

4.1 Open the notebook

Open:

ECG_final.ipynb

in Google Colab.

4.2 Mount Google Drive

The notebook uses:

from google.colab import drive


drive.mount('/content/drive')
4.3 Install WFDB

Run:

pip install wfdb

The notebook also uses Python libraries including:

NumPy
SciPy
PyTorch
scikit-learn
Matplotlib
WFDB
SQLite
4.4 Dataset

The project uses the MIT-BIH Arrhythmia Database from PhysioNet.

The notebook expects the local dataset archive under the configured Google Drive ECG directory.

The dataset itself is not included in this repository.

5. Execution Guide
Part A

Run the Part A notebook cells in order.

The workflow is:

Load MIT-BIH records
        ↓
Preprocess ECG signals
        ↓
Map annotations to AAMI classes
        ↓
Extract 270-sample beats
        ↓
Create record-level DS1 / DS2 split
        ↓
Create DS1 training / validation data
        ↓
Handle class imbalance
        ↓
Train 1D CNN
        ↓
Evaluate on DS2
Part B

Run the Part B cells in order:

B0 → B1 → B2 → B3

These cells:

save the baseline model;
initialize the SQLite database;
run baseline inference;
persist classification results.
Part C

Run the final Part C workflow in dependency order:

C0
 ↓
C1
 ↓
C4
 ↓
C2
 ↓
C3
 ↓
C5
 ↓
C6
 ↓
C6.5
 ↓
C7
 ↓
C8
C0

Inspects existing classified beats.

C1

Maintains the append-only reclassification audit trail.

C4

Creates the immutable locked DS2 copy and the separate DS1 adaptation dataset.

C2

Simulates reviewer corrections on the adaptation pool using reference annotations.

C3

Evaluates the configured re-learning trigger.

C5

Fine-tunes cnn_v1.0 using reviewer corrections and class-aware DS1 replay, producing cnn_v2.0.

C6

Compares cnn_v1.0 and cnn_v2.0 on the same locked DS2 dataset.

C6.5

Checks for catastrophic forgetting using per-class F1 and overall retention.

C7

Applies the promotion gate and updates the model registry.

C8

Reads the active model from the registry and performs inference on a new external ECG beat.

## 6. Summary of Results

### Locked DS2 evaluation

| Metric | cnn_v1.0 | cnn_v2.0 | Change |
|---|---:|---:|---:|
| Accuracy | 81.0794% | 83.8545% | +2.7751 pp |
| Macro F1 | 0.3031 | 0.3145 | +0.0114 |

### Per-class F1

| Class | cnn_v1.0 | cnn_v2.0 | Change |
|---|---:|---:|---:|
| N | 0.8940 | 0.9106 | +0.0166 |
| S | 0.0441 | 0.0463 | +0.0022 |
| V | 0.5774 | 0.6154 | +0.0380 |
| F | 0.0000 | 0.0000 | +0.0000 |
| Q | 0.0000 | 0.0000 | +0.0000 |

### Validation

The final validation run reported:

```text
Accuracy gate: PASS
Macro F1 gate: PASS

N: PASS
S: PASS
V: PASS
F: PASS
Q: PASS

Decision: PROMOTE cnn_v2.0
Active model: cnn_v2.0
### Catastrophic forgetting

The C6.5 analysis reported no evidence of catastrophic forgetting within the defined regression tolerance.

### External new-beat inference

The C8 demonstration used an external 270-sample ECG input.

The active model selected from the model registry was:

```text
cnn_v2.0
Input shape: (1, 1, 270)
Predicted class: N
Confidence: 0.9943## 7. Known Limitations

-  F and Q have F1 = 0.0000 for both model versions on the locked DS2 evaluation set. Their performance is reported as a class-sparsity/performance limitation. 
-  The reviewer workflow is a notebook-based simulation using reference annotations; it is not a separate clinician-facing web UI or REST API. 
-  The current implementation is notebook/Google-Colab based rather than packaged as a standalone production service. 
-  The repository currently contains the end-to-end notebook implementation rather than a separate modular Python package. 
-  The current inference demonstration uses an external ECG beat rather than demonstrating a separate production API. 

---

## 8. Reproducibility

The adaptive sampling uses a fixed random seed of:

```
42
```

The DS1/DS2 record split is fixed.

The final V1/V2 comparison uses the same locked DS2 evaluation set.

The model versions used by the workflow are:

```
cnn_v1.0
```

cnn\_v2.0

C8 reads the currently active model from the model registry rather than manually selecting a model version.

---

## 9. Assignment Coverage

The notebook implements the main requested workflow:

```
MIT-BIH ECG data
```

      ↓

Beat extraction + preprocessing

      ↓

AAMI classification

      ↓

Record-level DS1 / DS2 split

      ↓

1D CNN baseline

      ↓

Inference + SQLite persistence

      ↓

Simulated reviewer corrections

      ↓

Append-only audit trail

      ↓

Non-naive re-learning trigger

      ↓

Adaptive re-learning

      ↓

DS1 replay for forgetting mitigation

      ↓

Locked DS2 evaluation

      ↓

Catastrophic forgetting analysis

      ↓

Validation / promotion gate

      ↓

Active-model external inference

---

## 10. Notes

This project was developed and evaluated in Google Colab . The notebook contains the end-to-end implementation and the outputs used for the reported evaluation results.
