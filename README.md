## TurkishNewsProcessor

# News42NLP

News42NLP is a Python-based NLP pipeline for processing the "42bin_haber" Turkish news dataset. It implements text preprocessing, tokenization, TF-IDF vectorization, NMF topic modeling, and classification using libraries like NLTK, SciPy, and Hugging Face Tokenizers.

# Features

Text Preprocessing: Normalizes and cleans text, removing stopwords and applying lemmatization/stemming.

TF-IDF Vectorization: Builds sparse term-document matrices for efficient text representation.

Topic Modeling: Uses Non-Negative Matrix Factorization (NMF) to extract topics.

Search and Classification: Supports document search via cosine similarity and classification for labeled categories (e.g., magazin, saglik).

Optimized Performance: Leverages multiprocessing for faster processing of large datasets.

# Installation

Clone the repository:
'''bash
git clone https://github.com/yourusername/News42NLP.git
'''

Install dependencies:

'''bash
pip install -r requirements.txt
'''

Download NLTK resources:

'''pyhon
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
'''

# Usage

Place the "42bin_haber" dataset in the veri/42bin_haber/news/ directory.

Run the main processing script:

'''pyhon
python main_optimize.py <category> -p <processes> -d <experiment_name> -b <block_size>
'''

# Example:

'''pyhon
python main_optimize.py spor -p 4 -d deney1 -b 100
'''

Perform search:

'''pyhon
python main_arama.py
'''

Run classification:

'''pyhon
python siniflandirma.py
'''

# Files

main_optimize.py: Core script for preprocessing and TF-IDF computation.

main_arama.py: Implements document search using cosine similarity.

siniflandirma.py: Performs classification on labeled categories.

nmf.py: Applies NMF for topic modeling.

tokenizer_egitim.py: Trains a WordPiece tokenizer.

main_tokenizerhali.py: Processes data using a pre-trained tokenizer.

pencereleme.py: Implements text windowing for contextual analysis.

onisleme.py: Defines text preprocessing functions.

# Requirements

Python 3.8+
Libraries: nltk, scipy, numpy, tokenizers, pathlib

# Dataset

The project uses the "42bin_haber" dataset, a collection of Turkish news articles. Ensure the dataset is structured under veri/42bin_haber/news/ with category subfolders (e.g., spor, magazin, saglik).

# License

MIT License
