# Installation Guide

Use this guide to complete Phase 1: local setup, dependencies, and dataset placement.

## 1. Install Python

Install Python 3.9 or newer.

Check that PowerShell can find Python:

```powershell
python --version
```

If that does not work, try:

```powershell
py --version
```

## 2. Create the Environment

From the project root, run:

```powershell
.\scripts\setup_environment.ps1
```

Then activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Manual alternative:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt
```

## 3. Configure Optional Kaggle Access

If you want to download the dataset with the Kaggle API:

1. Copy `.env.example` to `.env`.
2. Add your Kaggle username and API key.
3. Or place your Kaggle token at the default Kaggle location.

Manual download is also fine.

## 4. Add the Dataset

Download the Kaggle dataset:

```text
https://www.kaggle.com/datasets/suddharshan/retail-price-optimization
```

Place the real file here:

```text
data/raw/retail_price.csv
```

The repository currently includes a tiny sample file at that path so code can be scaffolded before the real dataset is added.

## 5. Verify Setup

Run:

```powershell
python scripts/check_environment.py
pytest tests/
```

Your setup is ready when the environment check reports `Environment is ready.` and the tests pass.
