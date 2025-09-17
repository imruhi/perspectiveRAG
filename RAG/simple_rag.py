# Simple RAG
import os
import pickle

import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import load_dataset, cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"On device: {device}")
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


# question -> embed question ->
# search in predefined dataset -> give knowledge + question -> respond
class Query:
    """
    A class to define questions for RAG
    """

    def __init__(self, question, language):
        self.question = question
        self.language = language
        self.embedding = None


class SimpleRAG:
    def __init__(self, language_model_name, test, dataset_path='RAG_DB.pkl'):
        """
        A class to retrieve top K relevant chunks from given dataset and question and
        generate response using a LLM
        :param language_model_name: model to generate response from
        :param dataset_path: path where vector database of data is OR path where raw database is
        """
        self.lang_name = language_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.lang_name,
                                                       device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.lang_name,
                                                          quantization_config=nf4_config,
                                                          device_map=device)
        # TODO: hardcoded to dutch but need to change it
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2",
                                                   device=device)
        self.dataset_path = dataset_path
        self.questions = []

        # Indexing phase
        # vector_database = [(idx, chunk, embed), ...]
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'rb') as p:
                self.vector_database = pickle.load(p)
        else:
            self.vector_database = []
            print("No vector dataset found creating one")
            dataset = load_dataset(dataset_path.replace('.pkl', ''), test=test)
            self.create_vector_database(dataset)

    def embed_question(self, query: Query):
        """
        Todo: embed the question using the same model
        :param query: question to embed
        :return: set question embeddings
        """
        query.embedding = self.embedding_model.encode_query(query.question, convert_to_tensor=True)

    def chunk_texts(self, dataset):
        """
        Todo: chunk a given dataset into paragraphs?
        :param dataset: Dataset to chunk from (column "CleanedText")
        :return: A chunked dataset with all columns
        """
        text_dataset = dataset

        return text_dataset

    def create_vector_database(self, text_dataset, to_save=True):
        """
        Implementing a vector database
        :param to_save: save dataset or not
        :param text_dataset: hf Dataset object with chunked texts
        """
        chunked_database = self.chunk_texts(text_dataset)

        for idx, chunk in zip(chunked_database['ID'], chunked_database['CleanedText']):
            self.add_chunk_to_database(chunk, idx)
        if to_save:
            print(f"Added {len(chunked_database)} to vector database")
            with open(self.dataset_path, 'wb') as p:
                pickle.dump(self.vector_database, p)

    def add_chunk_to_database(self, chunk, idx, to_save=False):
        embedding = self.embedding_model.encode_document(chunk, convert_to_tensor=True)
        self.vector_database.append((idx, chunk, embedding))
        if to_save:
            with open(self.dataset_path, 'wb') as p:
                pickle.dump(self.vector_database, p)

    def get_chunks_from_database(self, idxs):
        chunks = []
        embeddings = []
        for i, chunk, embedding in self.vector_database:
            if i in idxs:
                chunks.append(chunk)
                embeddings.append(embedding)
        return chunks, embeddings

    def set_questions(self, questions: list, languages: list):
        """
        Given a list of questions and languages, set a list of Question objects
        :param languages: languages of questions
        :param questions: list of questions
        :return: set questions which will be used for RAG
        """
        for question, language in zip(questions, languages):
            self.questions.append(Query(question, language))

    def retrieve_top_idx(self, query: Query, top_k=3):
        """
        Return the k idx of most similar chunks from vector DB
        :param query: question asked
        :param top_k: # of most similar chunk's idx to be returned
        :return: list of idxs of length k and their similarities
        """
        sim_idx = []
        self.embed_question(query)

        for idx, chunk, embedding in self.vector_database:
            similarity = cosine_similarity(query.embedding, embedding)
            sim_idx.append((idx, similarity))
        sim_idx.sort(key=lambda x: x[1], reverse=True)
        return sim_idx[:top_k]

    def generate_response(self, top_k=3, max_new_tokens=200):
        decoded_outputs = []

        for query in self.questions:
            print(f"Retrieving {top_k} most similar chunks")
            retrieved_knowledge = self.retrieve_top_idx(query, top_k)

            chunks, embeddings = self.get_chunks_from_database([i for i, sim in retrieved_knowledge])

            for idx, similarity in retrieved_knowledge:
                print(f"    idx: {idx}, similarity: {similarity:.2f}")

            # TODO: currently only dutch
            prompt = f'''Je bent een behulpzame chatbot. 
            Gebruik de volgende contextfragmenten om de vraag te beantwoorden. Verzin geen nieuwe informatie:
            {chr(10).join([f' - {chunk}' for chunk in chunks])}
            '''

            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': query.question},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            print("Generating answers")
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
            decoded_output = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:])
            decoded_outputs.append(decoded_output)

        return decoded_outputs
