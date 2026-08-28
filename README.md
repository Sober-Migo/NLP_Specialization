# Natural Language Processing Specialization

**DeepLearning.AI · Coursera**  
*Taught by Younes Bensouda Mourri, Łukasz Kaiser, and Eddy Shyu (DeepLearning.AI)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Coursera](https://img.shields.io/badge/Coursera-Specialization-0056D2.svg)](https://www.coursera.org/specializations/natural-language-processing)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-required-red.svg)](https://git-lfs.com)

<p align="center">
  <img src="./NLP%20Certificate.png" alt="Natural Language Processing Specialization Certificate" width="700"/>
</p>

A complete collection of **lecture slides**, **optional labs**, **programming assignments**, **quizzes**, and **course materials** from the Natural Language Processing Specialization on Coursera.

> This repository contains my personal solutions and notes while completing the specialization. It is intended for educational reference only.

---

## Environment Setup & Installation

> **Why this section is at the top:**  
> While taking these courses I faced many environment, dependency, and large-file issues (missing packages, version conflicts, broken notebooks, large embeddings/datasets stored via Git LFS, zipped archives that need extraction, etc.). I had to figure everything out myself.  
> This repository is set up so you **don't have to go through the same pain**. The environment has been fully configured and tested — all notebooks across the four courses run without errors when you follow the steps below.

> **Tested environment:** Python **3.12** + `requirements.txt`  
> **Large files:** Git LFS is required (datasets, embeddings, `.zip` archives, `.h5`, `.npy`, large `.csv`, etc.)

### Important: Git LFS (Large File Storage)

This repository uses **Git LFS** for many large files (`.h5`, `.zip`, `.npy`, large `.csv`, `.mat`, `.mp4`, etc.).

- If you clone **without** Git LFS installed, you will only get tiny *pointer* files and the notebooks will fail.
- After installing Git LFS you must run `git lfs pull` (or use the setup script below).

**Install Git LFS** (one-time):

| Platform       | Command / Link                                      |
|----------------|-----------------------------------------------------|
| Windows / macOS| [https://git-lfs.com](https://git-lfs.com)          |
| Ubuntu/Debian  | `sudo apt install git-lfs`                          |
| Fedora         | `sudo dnf install git-lfs`                          |
| macOS (brew)   | `brew install git-lfs`                              |

Then:

```bash
git lfs install
```

### Recommended: Use the Interactive Setup Script

The easiest way to set everything up is to use the provided interactive script. It will:

1. Check / guide you on **Git LFS** and pull all large files
2. **Extract** every `.zip` archive found in the repo (and optionally delete the zips afterward to save space)
3. Let you choose **venv** or **Conda**
4. Install all packages from `requirements.txt`
5. Verify core packages and optionally launch Jupyter Lab

```bash
# 1. Clone the repository (Git LFS will download pointers; real files come later)
git clone https://github.com/Sober-Migo/NLP_Specialization.git
cd NLP_Specialization

# 2. Make sure Git LFS is installed, then run the setup script
python setup_env.py
```

The script works on **Windows**, **macOS**, and **Linux**.

---

### Manual Installation (Alternative)

#### 1. Clone + Git LFS

```bash
# Install Git LFS first (see table above), then:
git lfs install
git clone https://github.com/Sober-Migo/NLP_Specialization.git
cd NLP_Specialization
git lfs pull          # downloads all large files (datasets, embeddings, zips, models…)
```

#### 2. Extract the zip archives

```bash
# From the repository root (Linux / macOS / Git Bash)
find . -name "*.zip" -not -path "./.git/*" -not -path "./venv/*" | while read z; do
  echo "Extracting $z"
  unzip -o "$z" -d "$(dirname "$z")"
  # rm "$z"   # uncomment to delete zip after extraction
done
```

#### 3a. Using venv

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

jupyter lab   # or jupyter notebook
```

#### 3b. Using Conda

```bash
conda create -n nlp-specialization python=3.12 -y
conda activate nlp-specialization

pip install --upgrade pip
pip install -r requirements.txt

jupyter lab
```

> **Note on PyTorch / CUDA:**  
> `requirements.txt` pins `torch==2.6.0+cu124` (and matching torchvision/torchaudio).  
> If you do **not** have a compatible NVIDIA GPU + CUDA 12.4:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```

### Optional: Verify the environment

```bash
python -c "import numpy, pandas, sklearn, nltk, tensorflow, transformers; print('Core packages OK')"
python -c "import torch; print(f'PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"
```

---

## Table of Contents

- [Environment Setup & Installation](#environment-setup--installation)
- [About the Specialization](#about-the-specialization)
- [Repository Structure](#repository-structure)
- [Course Overview](#course-overview)
  - [Course 1 – Classification and Vector Spaces](#course-1--natural-language-processing-with-classification-and-vector-spaces)
  - [Course 2 – Probabilistic Models](#course-2--natural-language-processing-with-probabilistic-models)
  - [Course 3 – Sequence Models](#course-3--natural-language-processing-with-sequence-models)
  - [Course 4 – Attention Models](#course-4--natural-language-processing-with-attention-models)
- [Certificate](#certificate)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## About the Specialization

The **Natural Language Processing Specialization** is a program created by **DeepLearning.AI**. Across four courses you learn classic and modern NLP: sentiment analysis, vector spaces, probabilistic models, sequence models (RNNs/LSTMs), and attention / transformers for translation, summarization, and question answering.

**What you will learn:**

- Sentiment analysis with logistic regression and naïve Bayes
- Vector space models, word embeddings, PCA, and locality-sensitive hashing
- Autocorrect, n-gram language models, and part-of-speech tagging (HMMs)
- Sequence models: RNNs, LSTMs, GRUs, Siamese networks (sentiment, NER, duplicate questions)
- Attention, transformers, encoder–decoder models for machine translation, summarization, and chatbots

---

## Repository Structure

```text
NLP_Specialization/
├── C01 - Natural Language Processing with Classification and Vector Spaces/
│   ├── Base Dataset/
│   ├── Week 1/   Slides · Optional Labs · Assignment · Quiz
│   ├── Week 2/   Slides · Optional Labs · Assignment · Quiz
│   ├── Week 3/   Slides · Optional Labs · Assignment · Quiz
│   └── Week 4/   Slides · Optional Labs · Assignment · Quiz
├── C02 - Natural Language Processing with Probabilistic Models/
│   ├── Base Tokenizer/
│   ├── Week 1/ … Week 4/   Slides · Labs · Assignments · Quizzes
├── C03 - Natural Language Processing with Sequence Models/
│   ├── Week 1/ … Week 3/   Slides · Labs · Assignments · Quizzes
├── C04 - Natural Language Processing with Attention Models/
│   ├── Week 1/ … Week 3/   Slides · Labs · Assignments · Quizzes
├── NLP Certificate.png
├── NLP Certificate.pdf
├── requirements.txt
├── setup_env.py              # Interactive setup (Git LFS + zips + venv/Conda)
├── LICENSE
├── .gitattributes            # Git LFS tracking rules
├── .gitignore
└── README.md
```

> **Lecture slides** are in the `Slides/` folder inside each week (where present).

---

## Course Overview

### Course 1 – Natural Language Processing with Classification and Vector Spaces

| Week | Topics | Materials |
|------|--------|-----------|
| **Week 1** | Sentiment Analysis with Logistic Regression | [Slides](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%201/Slides) · [Optional Labs](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%201/Optional%20Labs) · [Assignment](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%201/Assignment) · [Quiz](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%201/Quiz) |
| **Week 2** | Sentiment Analysis with Naïve Bayes | [Slides](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%202/Slides) · Optional Labs · Assignment · Quiz |
| **Week 3** | Vector Space Models | [Slides](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%203/Slides) · Optional Labs · Assignment · Quiz |
| **Week 4** | Machine Translation and Document Search | [Slides](./C01%20-%20Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Week%204/Slides) · Optional Labs · Assignment · Quiz |

**Key Skills:** Logistic Regression · Naïve Bayes · Word Vectors · PCA · Locality-Sensitive Hashing · Approximate Nearest Neighbors

---

### Course 2 – Natural Language Processing with Probabilistic Models

| Week | Topics | Materials |
|------|--------|-----------|
| **Week 1** | Autocorrect & Minimum Edit Distance | [Week 1](./C02%20-%20Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%201) |
| **Week 2** | Part-of-Speech Tagging & Hidden Markov Models | [Week 2](./C02%20-%20Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%202) |
| **Week 3** | Autocomplete & Language Models (N-grams) | [Week 3](./C02%20-%20Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%203) |
| **Week 4** | Word Embeddings with Neural Networks | [Week 4](./C02%20-%20Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%204) |

**Key Skills:** Edit Distance · Viterbi · HMMs · N-gram LMs · Word2Vec-style embeddings

---

### Course 3 – Natural Language Processing with Sequence Models

| Week | Topics | Materials |
|------|--------|-----------|
| **Week 1** | Neural Networks for Sentiment Analysis | [Week 1](./C03%20-%20Natural%20Language%20Processing%20with%20Sequence%20Models/Week%201) |
| **Week 2** | Recurrent Neural Networks for Language Modeling | [Week 2](./C03%20-%20Natural%20Language%20Processing%20with%20Sequence%20Models/Week%202) |
| **Week 3** | LSTMs and Named Entity Recognition · Siamese Networks | [Week 3](./C03%20-%20Natural%20Language%20Processing%20with%20Sequence%20Models/Week%203) |

**Key Skills:** RNNs · LSTMs · GRUs · Named Entity Recognition · Siamese Networks · Question Duplicate Detection

---

### Course 4 – Natural Language Processing with Attention Models

| Week | Topics | Materials |
|------|--------|-----------|
| **Week 1** | Neural Machine Translation with Attention | [Week 1](./C04%20-%20Natural%20Language%20Processing%20with%20Attention%20Models/Week%201) |
| **Week 2** | Text Summarization with Transformer Models | [Week 2](./C04%20-%20Natural%20Language%20Processing%20with%20Attention%20Models/Week%202) |
| **Week 3** | Question Answering · Chatbots · Reformer | [Week 3](./C04%20-%20Natural%20Language%20Processing%20with%20Attention%20Models/Week%203) |

**Key Skills:** Attention · Transformers · Encoder–Decoder · T5 / BERT-style models · Summarization · QA · Reformer

---

## Certificate

<p align="center">
  <img src="./NLP%20Certificate.png" alt="Natural Language Processing Specialization Certificate" width="700"/>
</p>

You can also view the original PDF certificate:

**[NLP Certificate.pdf](./NLP%20Certificate.pdf)**

---

## Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Core | Python 3.12, NumPy, Pandas |
| Classical NLP | NLTK, scikit-learn, regex |
| Deep Learning | TensorFlow / Keras, PyTorch |
| Transformers | Hugging Face `transformers`, `tokenizers`, `sentencepiece` |
| Metrics / Utils | sacrebleu, seqeval, datasets |
| Visualization | Matplotlib, Seaborn |
| Environment | JupyterLab / Notebook |
| Large files | Git LFS |
| Others | See `requirements.txt` for the full pinned list |

---

## Disclaimer

This repository is created for **personal learning and educational purposes only**.

- The materials belong to **DeepLearning.AI** and **Coursera**.
- Solutions are shared to help fellow learners understand concepts, **not** to encourage academic dishonesty.
- Please attempt the assignments yourself first before referring to any solutions.
- Always respect Coursera’s Honor Code.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

### Useful Links

- [Natural Language Processing Specialization on Coursera](https://www.coursera.org/specializations/natural-language-processing)
- [DeepLearning.AI](https://www.deeplearning.ai/)
- [Git LFS](https://git-lfs.com)

---

⭐ If you find this repository helpful, consider giving it a star!
