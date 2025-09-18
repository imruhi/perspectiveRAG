import pickle

from utils import load_dataset
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from prompts import Prompt
from ragatouille import RAGPretrainedModel
import warnings
from os import path

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

warnings.filterwarnings("ignore")


class Query:
    """
    A class to define questions for RAG
    """

    def __init__(self, question, language):
        self.question = question
        self.language = language
        self.embedding = None
        self.retrieved_docs = None
        self.answer = None
        self.reranked_docs = None

    def set_embedding(self, emb):
        self.embedding = emb

    def set_answer(self, answer):
        self.answer = answer

    def set_retrieved_docs(self, retrieved_docs):
        self.retrieved_docs = retrieved_docs

    def set_reranked_docs(self, ranked_docs):
        self.reranked_docs = ranked_docs

    def get_reranked_doc_ids(self):
        retrieved_ids = self.get_retrieved_doc_ids()
        ids = []
        for doc in self.reranked_docs:
            ids.append(retrieved_ids[doc["result_index"]])
        return ids

    def get_retrieved_doc_ids(self):
        ids = []
        for doc in self.retrieved_docs:
            ids.append(doc.metadata['id'])
        return ids


class AdvancedRAG:
    def __init__(self, embedding_model_name, reader_model_name, cross_encoder_name, topics: list[int],
                 dataset_path='RAG_DB', temperature=0.2, max_new_tokens=300):
        # Init params
        self.dataset_path = dataset_path
        self.topics = topics
        self.embedding_model_name = embedding_model_name
        self.reader_model_name = reader_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cross_encoder_name = cross_encoder_name

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
        )

        self.vector_base = None
        self.knowledge_base = None
        # questions is a list of Query
        self.questions = []

        # set/load KB
        self.init_knowledge_base()

        # set reader model + tokenizer
        print("Setting reader model")
        self.reader_model = AutoModelForCausalLM.from_pretrained(self.reader_model_name, quantization_config=BNB_CONFIG)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reader_model_name)

        self.reader_llm = pipeline(
            model=self.reader_model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=self.temperature,  # Parameter to vary
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=self.max_new_tokens,
        )

        # set cross_encoder/reranker
        print("Setting reranker")
        self.reranker = RAGPretrainedModel.from_pretrained(self.cross_encoder_name)

    def init_knowledge_base(self):
        if path.exists(self.dataset_path + '_KB.pkl'):
            print("Loading KB")
            with open(self.dataset_path + '_KB.pkl', 'rb') as f:
                self.set_knowledge_base(pickle.load(f))
        else:
            # Load dataset
            ds = load_dataset(self.dataset_path, self.topics)
            # make KB, is a list of LangChain Docs (dataset is already chunked)
            print("Making KB")
            self.set_knowledge_base([
                LangchainDocument(page_content=doc["CleanedText"],
                                  metadata={"id": doc["ID"], "source": doc["Source"]})
                for idx, doc in ds.to_pandas().iterrows()
            ])
            with open(self.dataset_path+'_KB.pkl', 'wb') as f:
                pickle.dump(self.knowledge_base, f)

    def set_vector_store(self, vec_store):
        self.vector_base = vec_store

    def set_knowledge_base(self, kb):
        self.knowledge_base = kb

    def set_questions(self, queries: list, languages: list):
        """
        Given a list of questions and languages, set a list of Question objects and embed them
        :param languages: languages of questions
        :param queries: list of questions
        :return: set questions which will be used for RAG
        """
        for query, language in zip(queries, languages):
            self.questions.append(Query(query, language))
            # query.set_embedding(self.embedding_model.embed_query(query.question))

    def viz_chunk_embeds(self):
        # TODO: https://huggingface.co/learn/cookbook/en/advanced_rag
        pass

    def retrieve(self, top_k, query: Query):
        query.set_retrieved_docs(self.vector_base.similarity_search(query=query.question, k=top_k))

    def rerank(self, query: Query, retrieved_docs_text, top_k):
        retrieved_docs_text = self.reranker.rerank(query.question, retrieved_docs_text, k=top_k)
        query.set_reranked_docs(retrieved_docs_text)
        retrieved_docs_text = [doc["content"] for doc in retrieved_docs_text]
        return retrieved_docs_text

    def generate_prompt(self, retrieved_docs_text, query: Query):
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
        prompt_chat = Prompt(language=query.language, question=query.question, context=context).chat_prompt
        return self.tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)

    def prompt_model(self, brerank: bool = False, top_k=3, rerank_k=3):
        for query in self.questions:
            print("     => Retrieving documents...")
            self.retrieve(top_k, query)
            retrieved_docs_text = [doc.page_content for doc in query.retrieved_docs]

            if brerank:
                print("     => Reranking documents...")
                retrieved_docs_text = self.rerank(query, retrieved_docs_text, rerank_k)
            retrieved_docs_text = retrieved_docs_text[:top_k]

            print("     => Generating answer...")
            query.set_answer(self.reader_llm(self.generate_prompt(retrieved_docs_text, query))[0]["generated_text"])
