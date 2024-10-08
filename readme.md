# Annoy Index with Word2Vec - README

This repository demonstrates how to use **Annoy Index** with **Word2Vec** to perform approximate nearest neighbor search on text data. The text data is embedded using Word2Vec and indexed using Annoy to enable fast similarity searches.

## Table of Contents
1. [What is Annoy Index?](#what-is-annoy-index)
2. [What is Word2Vec?](#what-is-word2vec)
3. [How the System Works](#how-the-system-works)
4. [Installation](#installation)


## What is Annoy Index?

**Annoy** (Approximate Nearest Neighbors Oh Yeah) is a library that helps with searching for nearest neighbors using vector embeddings. It builds a forest of trees, where each tree is a spatial partition of the data, allowing for efficient similarity search.

### Features of Annoy:
- Fast approximate nearest neighbor search.
- Scales to a large number of items.
- Uses memory-mapped files, so it is memory-efficient.
- Supports a variety of distance metrics (e.g., angular, Euclidean).

Annoy is suitable for use cases where you need fast, approximate nearest neighbor lookups for large datasets, such as when searching for similar documents, words, or images.

## What is Word2Vec?

**Word2Vec** is a neural network-based technique that transforms words into continuous vector representations (embeddings). These embeddings capture semantic meaning, where words with similar meanings have vector representations that are close in the vector space.

### Features of Word2Vec:
- **Efficient training**: Word2Vec uses Skip-Gram or CBOW models to predict neighboring words in a sentence.
- **Contextual similarity**: Words that appear in similar contexts will have similar vector representations.
- **Unsupervised learning**: The model learns from unlabeled text data.

Word2Vec helps capture semantic similarities between words, which can then be used to perform operations like nearest neighbor searches.

## How the System Works

This project integrates Word2Vec for creating word embeddings and Annoy for efficient similarity search. Here's how it works:

1. **PDF/Text Preprocessing**: Text data from PDFs or other sources is split into manageable chunks.
2. **Word2Vec Embedding**: Each chunk of text is embedded into a vector using a Word2Vec model.
3. **Annoy Index Creation**: The vectors are added to an Annoy Index for fast nearest neighbor search.
4. **Querying the Index**: When querying the index, we convert the input query to a vector using Word2Vec, and Annoy returns the closest text chunks based on similarity.

## Installation

To get started, clone the repository and install the required packages:

```bash
git clone https://github.com/PrudhviChakravarthy/annoy_index
cd annoy_index
pip install -r requirements.txt
