import pickle
import numpy as np
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from typing import List, Optional

class PDFSplitter:
    def __init__(self, chunk_size=400, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    def split_text(self, text: str) -> List[str]:
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        return splitter.split_text(text)

class PDFReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def extract_text_by_page(self) -> List[str]:
        reader = PdfReader(self.file_path)
        pages = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pages.append(page.extract_text())
        return pages

class Word2VecEmbedder:
    def __init__(self, model: Optional[Word2Vec] = None):
        self.model = model if model else Word2Vec(min_count=1, vector_size=100, workers=4)
    def fit(self, corpus: List[List[str]]):
        self.model.build_vocab(corpus)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=10)
    def get_vector(self, word: str) -> np.ndarray:
        """Retrieve the vector for a given word. If OOV, return a zero vector."""
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            return np.zeros(self.model.vector_size)   

    def get_mean_vector(self, text: str) -> np.ndarray:
        words = text.split()  
        vectors = [self.get_vector(word) for word in words if np.any(self.get_vector(word) != 0)]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)  
        else:
            return np.zeros(self.model.vector_size)  

class AnnoyVectorStore:
    def __init__(self, vector_size: int, n_trees: int = 10):
        self.index = AnnoyIndex(vector_size, 'angular')
        self.vector_size = vector_size
        self.n_trees = n_trees
        self.embedded_texts = []
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[str]):
        for idx, vector in enumerate(vectors):
            self.index.add_item(idx, vector)
            self.embedded_texts.append(metadata[idx])
    def build_index(self, n_jobs: int = -1):
        self.index.build(self.n_trees, n_jobs=n_jobs)
    def save(self, index_file: str, pkl_file: str):
        self.index.save(index_file)
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.embedded_texts, f)
    def load(self, index_file: str, pkl_file: str):
        self.index.load(index_file)
        with open(pkl_file, 'rb') as f:
            self.embedded_texts = pickle.load(f)
    def query(self, vector: np.ndarray, top_n: int = 5):
        print(vector, top_n)
        result_indices = self.index.get_nns_by_vector(vector, top_n, include_distances=True)
        print(result_indices)
        results = [(self.embedded_texts[i], dist) for i, dist in zip(*result_indices)]
        print(results)
        return results

class PDFProcessor:
    def __init__(self, pdf_splitter: PDFSplitter, word2vec_embedder: Word2VecEmbedder, vector_store: AnnoyVectorStore):
        self.pdf_splitter = pdf_splitter
        self.word2vec_embedder = word2vec_embedder
        self.vector_store = vector_store
    def process_and_add_pdf(self, pdf_text: List[str]):
        all_chunks = []
        for page_text in pdf_text:
            chunks = self.pdf_splitter.split_text(page_text)
            all_chunks.extend(chunks)
        tokenized_chunks = [chunk.split() for chunk in all_chunks]  
        self.word2vec_embedder.fit(tokenized_chunks)  
        vectors = [self.word2vec_embedder.get_mean_vector(chunk) for chunk in all_chunks]
        self.vector_store.add_vectors(vectors, all_chunks)
    def build_and_save_index(self, n_jobs: int, ann_file: str, pkl_file: str):
        self.vector_store.build_index(n_jobs=n_jobs)
        self.vector_store.save(ann_file, pkl_file)
    def load_index_and_query(self, query: str, ann_file: str, pkl_file: str, top_n: int = 5):
        self.vector_store.load(ann_file, pkl_file)
        query_vector = self.word2vec_embedder.get_mean_vector(query)
        return self.vector_store.query(query_vector, top_n=top_n)
if __name__ == "__main__":
    pdf_splitter = PDFSplitter(chunk_size=40, overlap=5)
    word2vec_embedder = Word2VecEmbedder()
    vector_store = AnnoyVectorStore(vector_size=100, n_trees=10)
    processor = PDFProcessor(pdf_splitter, word2vec_embedder, vector_store)
    pdf_reader = PDFReader(file_path="what.pdf")
    # pdf_pages = pdf_reader.extract_text_by_page()
    # processor.process_and_add_pdf(pdf_pages)
    # processor.build_and_save_index(n_jobs=-1, ann_file='vector_store.ann', pkl_file='vector_store.pkl')
    query = str(input("ask query?"))
    results = processor.load_index_and_query(
        query=query,
        ann_file='vector_store.ann',
        pkl_file='vector_store.pkl',
        top_n=10
    )
    for result in results:
        print(f"Text: {result[0]}, Distance: {result[1]}")
