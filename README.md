# DECKER

DECKER (Double Check with Heterogeneous Knowledge for Commonsense Fact Verification) classifies claims using both ConceptNet's structured relations and facts retrieved from Wikipedia. The pipeline grounds concepts in each claim, retrieves supporting passages, builds a heterogeneous graph, and trains a DeBERTa-based classifier.

![DECKER architecture](pics/overview.png)

## Requirements

- Python 3.10 or later
- Git Bash or WSL for the shell scripts on Windows
- A CUDA-capable GPU is recommended for training. Retrieval falls back to CPU, but processing the full Wikipedia collection on CPU can be very slow.
- Internet access to download pretrained Hugging Face models and the external resources below

Create an isolated environment from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Choose a PyTorch build that matches your CUDA setup from [the PyTorch install selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements if GPU training is needed.

## Data And Models

The repository does not include datasets, pretrained weights, or the large Wikipedia index. Keep downloaded files under `data/`; that directory is excluded from Git.

1. Download CREAK and CSQA2 from their project pages. Place the files at these exact paths:

	- CREAK: `data/creak/train.json`, `dev.json`, `test_without_labels.json`, and `contrast_set.json`
	- CSQA2: `data/csqa2/CSQA2_train.json`, `CSQA2_dev.json`, and `CSQA2_test_no_answers.json`

	Sources: [CREAK](https://github.com/yasumasaonoe/creak/tree/main/data/creak) and [CSQA2](https://github.com/allenai/csqa2/tree/master/dataset).

2. Run `download_rawdata_resources.sh` from the repository root in Git Bash or WSL. It downloads the Wikipedia passages and Contriever embeddings, plus the raw ConceptNet archive.

3. The raw ConceptNet archive is not enough for this code. Before preprocessing, provide the derived files `data/cpnet/concept.txt`, `data/cpnet/matcher_patterns.json`, and `data/cpnet/conceptnet.en.pruned.graph`. These are generated ConceptNet resources and are not created by the download script.

4. The training launcher downloads `microsoft/deberta-v3-large` from Hugging Face on its first run. Retrieval downloads `facebook/contriever` as needed.

## Run

Activate the virtual environment, open Git Bash (or WSL) at the repository root, and run preprocessing:

```bash
cd data_preprocess
bash data_preprocess.sh
```

Preprocessing has three stages: concept grounding, Wikipedia retrieval, and graph construction. It writes JSONL files under `data/creak/` and `data/csqa2/`. Windows runs grounding and graph construction sequentially; Linux/WSL uses multiprocessing.

Train the default CREAK experiment from the repository root:

```bash
bash confact_train.sh
```

The launcher uses `microsoft/deberta-v3-large`, five epochs, and the CREAK graph files. To train on CSQA2, change `DATA="creak"` to `DATA="csqa2"` in `confact_train.sh`. Training and evaluation metrics are written under `saved_outputs_my/`; test and CREAK contrast predictions are saved as `test_predictions.txt` and `contra_predictions.txt`.

## Verify

Check Python syntax without downloading the datasets or model:

```powershell
python -m compileall -q run_confact.py dataset.py model data_preprocess
```

A full training screenshot is not included yet: this checkout has no datasets, generated ConceptNet graph files, or model cache, so a successful training run could not be captured here. The architecture image above is a project diagram, not a run result. After the prerequisites are in place, capture the terminal showing completed training and prediction output, then add it under `pics/` and link it here.

## Push To Your GitHub

Create an empty repository on GitHub, then replace the starter project's `origin` remote with your repository. Run these commands from the project root, substituting your GitHub username and repository name:

```bash
git remote rename origin upstream
git remote add origin https://github.com/<your-username>/<your-repository>.git
git status
git add .
git commit -m "Fix and document DECKER project"
git push -u origin main
```

The `.gitignore` excludes datasets, checkpoints, virtual environments, and caches. Do not force-add large data or model files; link to their original sources instead.



## Screenshots(Output)

<img width="1600" height="980" alt="WhatsApp Image 2026-11-05 at 7 44 40 PM" src="https://github.com/user-attachments/assets/fd00b590-43d3-4fe0-a0fd-9f1bc3159eae" />
<img width="1600" height="978" alt="WhatsApp Image 2026-11-05 at 8 19 43 PM" src="https://github.com/user-attachments/assets/6c2e18b4-e807-4295-a979-93b1cc935e27" />
<img width="982" height="598" alt="WhatsApp Image 2026-11-05 at 7 39 32 PM" src="https://github.com/user-attachments/assets/581b0dd4-582b-4117-9191-631d1e4a9182" />

