import pickle

from advanced_rag import AdvancedRAG
import pandas as pd
from multiprocessing import freeze_support
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

pd.set_option("display.max_colwidth", None)

vec_save = "faiss_index.pkl"

# rag params
model_name = "Qwen/Qwen2-7B-Instruct"
cross_encoder_name = "colbert-ir/colbertv2.0"  # "cross-encoder/ms-marco-MiniLM-L6-v2"
embedding_model_name = "all-MiniLM-L6-v2"
max_new_tokens = 300
temperature = 0.3
dataset_path = "data/RAG_DB"

# generation params
rerank = True
top_k = 5

# query params
questions = [
    "Wie was Napoleon Bonaparte?",
    "Was Napoleon Bonaparte een held?",
    "Was Napoleon Bonaparte een tiran?",
]
languages = [
    "nl",
    "nl",
    "nl",
]


# sRAG = SimpleRAG(language_model_name=model_name, test=True, dataset_path=dataset_path)
# sRAG.set_questions(questions, languages)
# answers = sRAG.generate_response(top_k=top_k, max_new_tokens=max_new_tokens)
def main():
    aRag = AdvancedRAG(embedding_model_name=embedding_model_name, max_new_tokens=max_new_tokens,
                       reader_model_name=model_name, cross_encoder_name=cross_encoder_name,
                       dataset_path=dataset_path, temperature=temperature)

    # vector database init based on cosine
    print("Making VD")
    # in main due to synchronity issues
    aRag.vector_base = FAISS.from_documents(
        aRag.knowledge_base, aRag.embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    # TODO: better save and load?
    with open(vec_save, 'wb') as f:
        pickle.dump(aRag.vector_base, f)

    aRag.set_questions(questions, languages)
    aRag.prompt_model(brerank=rerank, top_k=top_k)

    for query in aRag.questions:
        print(f"Query: {query.question}")
        print(f"Answer: {query.answer}")


if __name__ == "__main__":
    freeze_support()
    main()
