# ASA-PS Prediction from Clinical Text

This repository contains code for experiments and figures for the manuscript: 

Preprint: [Prediction of American Society of Anesthesiologists Physical Status Classification from Preoperative Clinical Text Narratives Using Natural Language Processing]((https://www.medrxiv.org/content/10.1101/2023.02.03.23285402v1))
Philip Chung, Christine T. Fong, Andrew M. Walters, Meliha Yetisgen, Vikas N. O'Reilly-Shah
medRxiv 2023.02.03.23285402; doi: https://doi.org/10.1101/2023.02.03.23285402

## Task

In this project, we aim to show how free-text clinical narratives from medical notes and free-text from surgery booking fields can be used for perioperative risk stratification.  We use the American Society of Anesthesiologists Physical Status Classification (ASA-PS) score as a surrogate for a patient's underlying medical complexity and general state of health.

### Model Performance

We examine the use of 4 different model types to perform the prediction task.  Each model type uses different paradigms for text representation and classification.

* [__Random Forest__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): Ensemble of decision trees operating on unigram & bigram text features.
* [__Support Vector Machine__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html): Separates output classes with hyperplanes using unigram & bigram text features.
* [__fastText__](https://fasttext.cc/): Shallow neural network classification using contextually learned word vectors.
* [__BioClinicalBERT__](https://github.com/EmilyAlsentzer/clinicalBERT): Transformer deep neural network language model for text sequence classification.

Baseline Models for comparison:

* [__Random Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html): Negative control baseline comparison model that randomly picks an ASA-PS score without using any input features.
* [__Age Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): Clinical baseline comparison model that uses age to predict ASA-PS via simple logistic regression model.  As patients age, they tend to have more medical conditions and comorbidities which tend to increase the ASA-PS score.
### Clinical Note Sections

Clinical notes are often written with distinct sections which provide conceptually different pieces of information.  We also explore the performance of predicting ASA-PS using different snippets of text from a Pre-Anesthesia note:

* __Description__: Description/name of planned surgery (or surgeries).  Often present in the note, but is more reliably present from anesthetic record and extracted from the anesthetic record.
* __Diagnosis__: Diagnosis at time of surgery booking.  Often present in the note, but is more reliably present from anesthetic record and extracted from the anesthetic record.
* __History of Present Illness (HPI)__: Narrative that summarizes patient's medical status and why they are having surgery.
* __Review of Systems (ROS)__: Semi-structured list or narrative of medical conditions, issues, concerns, organized by organ systems.
* __Past Medical Surgical History (PMSH)__: Combination of Past Medical History (list of medical conditions) and Past Surgical History (list of surgeries).
* __Medications (Meds)__: List of medications patient is taking.  Typically automatically generated from other tables in electronic health record.
* __Note Text (Note)__: Full Pre-Anesthesia note raw text from which the above sections have been extracted.  Also includes other note sections.
* __Truncated Note Text (Note512)__: Same as "Note Text", but truncated to first 512 tokens from WordPiece tokenizer.  For fair comparison of other models against BioClinicalBERT model which is limited to sequence of 512 input tokens.

### ASA Physical Status Classification

The [American Society of Anesthesiologists Physical Status Classification (ASA-PS)](https://www.asahq.org/standards-and-guidelines/asa-physical-status-classification-system) is extracted from the anesthetic record and thus reflects the ASA-PS assigned by the anesthesiologist from their assessment of the patient on the day of surgery.

Unlike other medical scoring systems which have strict definitions or diagnostic criteria, there is no strict criteria for each ASA-PS class.  Rather the American Society of Anesthesiologists provides examples for each ASA-PS class.  In effect, the ASA-PS constitute classification bins with fuzzy boundaries.  There is known inter-rater disagreement between human anesthesiologists.  Nonetheless, the ASA-PS has been found in many studies to be an independent predictor of morbidity and mortality in surgical patients.

In our prediction task, we make several modifications for practical reasons:

* ASA VI (organ donation procurement cases) are excluded
* ASA V cases are rare, so they are combined with ASA IV to create a compound "ASA IV-V" class.
* Emergency modifier "E" is ignored for our analysis.  It is often missing for emergency cases in the ground truth data.

## Project Configuration

### Repository Structure

This directory is the project root.  MLproject file defines entry points.  Call all entry points with this directory as current working directory.  A brief description of directory contents:

* __bert__: BioClinicalBERT model training, evaluation, interpretation scripts
* __conf__: hydra configurations for scripts
* __dummy__: Random "Dummy" model baseline (negative control)
* __fasttext__: fastText model training, evaluation
* __misc__: utility scripts
* __rf__: Random Forest model training, evaluation
* __src__: shared library across all model training & evaluation scripts
* __svm__: Support Vector Machine model training, evaluation

### Data

Our dataset contains identifiable patient information and is not publically available.  Ultimately note text and sections are extracted, preprocessed into a table with the following columns:

* index: example number in dataset
* input_text: cleaned & preprocessed input text.  Selection of a specific text snippet/note section is performed as part of the preprocessing.
* input_ids: created by tokenizer, only used by BioClinicalBERT
* token_type_ids: created by tokenizer, only used by BioClinicalBERT
* attention_mask: created by tokenizer, only used by BioClinicalBERT
* asa_label: output class label (I, II, III, IV-V)
* emergency_label: "E" designation (True vs. False)

This data table is used for model training and evaluation.
### Python Environment

Python environment is managed with:

* [`pyenv`](https://github.com/pyenv/pyenv): manage python version globally on system and locally in project folder
* [`poetry`](https://python-poetry.org/): package management and dependency resolution.  Package dependencies are specified in `pyproject.toml`, which is used to create a `poetry.lock` file which contains fully resolved dependencies that constitute a reproducible environment.  If a `poetry.lock` file is present, then `poetry install` will simply install all packages from the lock file.

```sh
# Navigate to repo root
cd nlp-asa-prediction
# Installation & Configuration script for pyenv and poetry on a new Azure Compute Instance
# Make sure to change USERNAME={your_user_name_on_machine}
environment/pyenv_poetry_setup.sh
# This bash script also exports all installed packages as a requirements.txt

# Activate virtual environment managed by poetry.
poetry shell
# now you can run any command within the environment that uses project dependencies

# See location of poetry virtual environment & python kernel
poetry env info 
```

The `poetry shell` command automatically creates a virtual environment which encapsulates all the project dependencies.

To use this python kernel in Jupyter Notebooks:

```sh
# Create detectable python kernel called `nlp-asa-prediction`
poetry run python -m ipykernel install --user --name nlp-asa-prediction
```

### Dependencies

In addition to Azure Machine Learning, this project installs several external dependencies:

* __[sci-kit learn](https://scikit-learn.org/stable/)__: random forest and support vector machine
* __[fastText](https://fasttext.cc/)__: fasttext word vector models
* __[huggingface transformers](https://huggingface.co/docs/transformers/index)__: BioClinicalBERT and BioDischargeSummaryBERT models
* __[lightning](https://www.pytorchlightning.ai/)__: training pytorch models
* __[torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)__: model evaluation metrics
* __[mlflow](https://mlflow.org/docs/latest/index.html)__: experiment tracking, logging
* __[ray-tune](https://docs.ray.io/en/latest/tune/index.html)__: hyperparameter tuning
* __[FLAML](https://microsoft.github.io/FLAML/)__: hyperparameter search using BlendSearch
* __[hydra](https://hydra.cc/)__: script config management and command-line options

### Notebooks

`notebooks` directory contain jupyter notebooks for model evaluation and figure generation for manuscript.

### Entry Points

Entry Points to run train, tune, evaluation scripts are documented in the `MLproject` file.

```sh
# Train BioClincalBERT Model
python models/bert/train.py
# Results are in ./outputs/...
```

### Hydra Configurations

This project uses Hydra for configuration management.  All configurations are stored in `conf` directory, which are referenced by the entry point scripts.  These configurations are composable and can be overridden at the configuration group level or individual parameter level.

```sh
# View default hydra configuration for training job without running script
python models/rf/train.py --cfg job --resolve
```

```sh
# Train Random Forest, Override default config to specify different configuration group for the model and run the model in debug mode
python models/rf/train.py model=rf_debug
```

```sh
# Train Random Forest, Override default config to specify individual model parameters
python models/rf/train.py model.params.n_estimators=10 model.params.n_jobs=10
```

### Task Selection

```sh
# Various Tasks Trained using BioClinicalBERT Model
# Defaults to `hpi-asa` task
python models/bert/train.py
# Explicitly select task={input_clinical_note_section_text}-{output_asa_varible}
python models/bert/train.py task=procedure-asa
python models/bert/train.py task=diagnosis-asa
python models/bert/train.py task=hpi-asa
python models/bert/train.py task=ros-asa
python models/bert/train.py task=pmsh-asa
python models/bert/train.py task=meds-asa
python models/bert/train.py task=note-asa
python models/bert/train.py task=note512-asa
```

### MLFlow Runs

When each entrypoint script is invoked, Hydra automatically creates an `outputs` subdirectory on the local file system in which logs and artifacts produced by the run are saved.  

This project also uses MLflow on Azure Machine Learning for experiment and run tracking, metrics logging, model artifact logging.  Each run is organized in groups called "experiments".  Trained models are uploaded to the model registry.

### Hyperparameter Tuning

Each hyperparameter tuning job creates a parent run in MLflow and each tuning trial corresponding to a set of hyperparameters is logged as a nested child run.

```sh
# Hyperparameter Tune BERT Model (defaults to task=hpi-asa)
python models/bert/tune.py

# Hyperparameter Tune BERT for different tasks
python models/bert/tune.py task=note512-asa

# Hyperparameter Tune BERT Model, set number of trials, relax stopping criteria
python models/bert/tune.py tune.params.num_trials=20, tune.stopper.top=5 tune.stopper.patience=1 tune.stopper.std=0.01

# Export copy of data to trial run output directory and use that copy of data to train.
python models/bert/tune.py general.export_dataset=true
```

## Hyperparameter Tuning Configurations

```sh
# Tune BioClinicalBERT
python models/bert/tune.py task=procedure-asa mlflow.suffix=_tune1
python models/bert/tune.py task=diagnosis-asa mlflow.suffix=_tune1
python models/bert/tune.py task=hpi-asa mlflow.suffix=_tune1
python models/bert/tune.py task=ros-asa mlflow.suffix=_tune1
python models/bert/tune.py task=pmsh-asa mlflow.suffix=_tune1
python models/bert/tune.py task=meds-asa mlflow.suffix=_tune1
python models/bert/tune.py task=note-asa mlflow.suffix=_tune1
python models/bert/tune.py task=note512-asa mlflow.suffix=_tune1

# Tune Random Forest
python models/rf/tune.py task=procedure-asa mlflow.suffix=_tune1
python models/rf/tune.py task=diagnosis-asa mlflow.suffix=_tune1
python models/rf/tune.py task=hpi-asa mlflow.suffix=_tune1
python models/rf/tune.py task=ros-asa mlflow.suffix=_tune1
python models/rf/tune.py task=pmsh-asa mlflow.suffix=_tune1
python models/rf/tune.py task=meds-asa mlflow.suffix=_tune1
python models/rf/tune.py task=note-asa mlflow.suffix=_tune1
python models/rf/tune.py task=note512-asa mlflow.suffix=_tune1

# Tune Support Vector Machine
python models/svm/tune.py task=procedure-asa mlflow.suffix=_tune1
python models/svm/tune.py task=diagnosis-asa mlflow.suffix=_tune1
python models/svm/tune.py task=hpi-asa mlflow.suffix=_tune1
python models/svm/tune.py task=ros-asa mlflow.suffix=_tune1
python models/svm/tune.py task=pmsh-asa mlflow.suffix=_tune1
python models/svm/tune.py task=meds-asa mlflow.suffix=_tune1
python models/svm/tune.py task=note-asa mlflow.suffix=_tune1
python models/svm/tune.py task=note512-asa mlflow.suffix=_tune1

# Tune FastText
python models/fasttext/tune.py task=procedure-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=diagnosis-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=hpi-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=ros-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=pmsh-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=meds-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=note-asa mlflow.suffix=_tune1
python models/fasttext/tune.py task=note512-asa mlflow.suffix=_tune1
```

### Evaluation Configurations

```sh
# Evaluate Random Forest
python models/rf/train_evaluate.py task=procedure-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=diagnosis-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=hpi-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=ros-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=pmsh-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=meds-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=note-asa mlflow.suffix=_tune1
python models/rf/train_evaluate.py task=note512-asa mlflow.suffix=_tune1

# Evaluate Support Vector Machine
python models/svm/train_evaluate.py task=procedure-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=diagnosis-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=hpi-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=ros-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=pmsh-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=meds-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=note-asa mlflow.suffix=_tune1
python models/svm/train_evaluate.py task=note512-asa mlflow.suffix=_tune1

# Evaluate FastText
python models/fasttext/train_evaluate.py task=procedure-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=diagnosis-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=hpi-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=ros-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=pmsh-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=meds-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=note-asa mlflow.suffix=_tune1
python models/fasttext/train_evaluate.py task=note512-asa mlflow.suffix=_tune1

# Evaluate BioClinicalBERT
python models/bert/train_evaluate.py task=procedure-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=diagnosis-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=hpi-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=ros-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=pmsh-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=meds-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=note-asa mlflow.suffix=_tune1
python models/bert/train_evaluate.py task=note512-asa mlflow.suffix=_tune1

# Evaluate Baseline Models (Random Classifier & Age Classifier)
python models/baseline/train_evaluate.py task=note512-asa
```

### SHAP Model Explanations

Shapley values can be computed and overlaid on input text to understand how individual features (words) contribute to predicting each output class.  Computing Shapley values takes a long time and is compute intensive.

```sh
# SHAP Explanations for BioClinicalBERT
python models/interpret/shap_clinicalbert.py task_name="note-asa"
python models/interpret/shap_clinicalbert.py task_name="note512-asa"
# SHAP Explanations for FastText
python models/interpret/shap_fasttext.py task_name="note-asa"
python models/interpret/shap_fasttext.py task_name="note512-asa"
```

### Debugging Configurations

Debug configurations turn off mlflow model artifact logging, log runs under experiment `debug`, adjust model settings to enable them to train faster, increases logging intervals and log output from scripts.

```sh
# Debug Random Forest
python models/rf/train.py mlflow=debug model=rf_debug
python models/rf/tune.py mlflow=debug model=rf_tune_debug tune=cpu_debug
# Debug Support Vector Machine
python models/svm/train.py mlflow=debug model=svm_debug
python models/svm/tune.py mlflow=debug model=svm_tune_debug tune=cpu_debug
# Debug Fasttext
python models/fasttext/train.py mlflow=debug
python models/fasttext/tune.py mlflow=debug tune=cpu_debug
# Debug BioclinicalBERT
python models/bert/train.py mlflow=debug lightning=debug model=bioclinicalbert
python models/bert/tune.py mlflow=debug lightning=debug tune=gpu_debug model=bioclinicalbert_tune
# Debug DischargeSummaryBERT
python models/bert/train.py mlflow=debug lightning=debug model=dischargesummarybert datamodule=dischargesummarybert
python models/bert/tune.py mlflow=debug lightning=debug tune=gpu_debug model=dischargesummarybert_tune datamodule=dischargesummarybert
```

## Miscellaneous

### Generate Cache Files for all Tasks

This script generates cache files for dataset preprocessing for all tasks in default dataset location, which is at `project_dir/data/.../processed/hpi-pmsh-ros-meds-asa`.  Future scripts that load dataset from default location will reuse the cached files.

```sh
# Preprocess dataset for all tasks & save cache
# also generates training dataset in fasttext format
python misc/make_datasets_cache.py
```

### Clean Up Cache Files in Datasets

After dataset is created, it is preprocessed for each task prior to being used to train with a model.  These caches are backed by Apache Arrow using the huggingface datasets library.  This cache can be cleared using a script.

```sh
# Clear cache for default dataset at default path
python misc/clear_datasets_cache.py
# Clear cache for specific dataset name and path location
python misc/clear_datasets_cache.py +name=hpi-pmsh-ros-meds-asa +="~/project/data"
```

### Model Deletion

Models stored in model registry can be programatically deleted en masse.

```sh
# Delete model name "hpi-asa-random-forest" version 5, 6, 7.
python misc/delete_models.py +name=hpi-asa-random-forest +version_start=5 +version_end=7
# Models that don't exist in the range between version_start and version_end are skipped.
```
