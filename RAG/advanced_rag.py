import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import pickle
from utils import load_dataset
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from prompts import Prompt
from ragatouille import RAGPretrainedModel
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
        self.answers = None
        self.reranked_docs = None
        self.baseline_answers = None

    def set_embedding(self, emb):
        self.embedding = emb

    def set_answers(self, answers):
        self.answers = answers

    def set_retrieved_docs(self, retrieved_docs):
        self.retrieved_docs = retrieved_docs

    def set_reranked_docs(self, ranked_docs):
        self.reranked_docs = ranked_docs

    def set_baseline_answers(self, baselines):
        self.baseline_answers = baselines

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
                 dataset_path='RAG_DB', temperature=0.2, max_new_tokens=300, bwiki="False"):
        # Init params
        self.dataset_path = dataset_path
        self.topics = topics
        self.embedding_model_name = embedding_model_name
        self.reader_model_name = reader_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cross_encoder_name = cross_encoder_name
        self.language = None
        self.vector_base = None
        self.knowledge_base = None
        # questions is a list of Query
        self.questions = []
        self.bwiki = True if bwiki == "True" else False

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
        )

        # set reader model + tokenizer
        print("Setting reader model", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reader_model_name)
        self.reader_model = AutoModelForCausalLM.from_pretrained(self.reader_model_name,
                                                                 # quantization_config=BNB_CONFIG,
                                                                 dtype="auto", device_map="auto")

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
        print("Setting reranker", flush=True)
        self.reranker = RAGPretrainedModel.from_pretrained(self.cross_encoder_name)

    def prepare(self):
        # set/load KB
        self.init_knowledge_base()

    def init_knowledge_base(self):
        add_on = "withwiki" if self.bwiki else "withoutwiki"
        new_path = self.dataset_path + '_' + self.language + add_on + '_KB.pkl'
        if path.exists(new_path):
            print("Loading KB", flush=True)
            with open(new_path, 'rb') as f:
                self.set_knowledge_base(pickle.load(f))
        else:
            # Load dataset
            ds = load_dataset(self.dataset_path, self.topics, self.language, self.bwiki)
            # make KB, is a list of LangChain Docs (dataset is already chunked)
            print("Making KB", flush=True)
            self.set_knowledge_base([
                LangchainDocument(page_content=doc["CleanedText"],
                                  metadata={"id": doc["ID"], "source": doc["Source"]})
                for idx, doc in ds.to_pandas().iterrows()
            ])
            with open(new_path, 'wb') as f:
                pickle.dump(self.knowledge_base, f)

    def set_language(self, language):
        self.language = language

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
        # TODO: retrieve documents based on source? need to add filter={"source": source} param
        query.set_retrieved_docs(self.vector_base.max_marginal_relevance_search(query=query.question, k=top_k,
                                                                                fetch_k=2 * top_k, lambda_mult=0.5))

    def rerank(self, query: Query, retrieved_docs_text, top_k):
        retrieved_docs = self.reranker.rerank(query.question, retrieved_docs_text, k=top_k)
        query.set_reranked_docs(retrieved_docs)
        reranked_docs_text = [doc["content"] for doc in retrieved_docs]
        return reranked_docs_text

    def generate_prompt(self, retrieved_docs_text, query: Query):
        context = ""
        context += "".join([f"\nDocument {str(i)}:::" + doc for i, doc in enumerate(retrieved_docs_text)])
        prompt_chat = Prompt(language=query.language, question=query.question, context=context).chat_prompt
        return self.tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def prompt_model_baseline(self, repeat):
        for query in self.questions:
            baseline = []
            prompt = self.generate_prompt([], query)
            answers = self.reader_llm([prompt] * repeat)
            baseline.extend([i["generated_text"] for x in answers for i in x])
            query.set_baseline_answers(baseline)

    def prompt_model(self, brerank: bool = False, top_k=3, rerank_k=3, repeat=3):
        """
        :param repeat: ONLY REPEAT GENERATION NOT RETRIEVAL AND RERANKING
        """

        all_prompts = []

        for query in self.questions:
            self.retrieve(top_k, query)
            retrieved_docs_text = [doc.page_content for doc in query.retrieved_docs]
            retrieved_docs_text = retrieved_docs_text[:top_k]

            if brerank:
                retrieved_docs_text = self.rerank(query, retrieved_docs_text, rerank_k)
                retrieved_docs_text = retrieved_docs_text[:rerank_k]

            prompt = self.generate_prompt(retrieved_docs_text, query)
            all_prompts.append(prompt)

        answers = self.reader_llm(all_prompts)
        answers = [i["generated_text"].replace('\n', ' ') for x in answers for i in x]

        for query, answer in zip(self.questions, answers):
            query.set_answers(answer)

        print("\n", flush=True)
