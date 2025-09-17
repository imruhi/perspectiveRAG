from simple_rag import Query, SimpleRAG

model_name = "meta-llama/Llama-3.1-8B-Instruct"
top_k = 3
max_new_tokens = 200
dataset_path = "data/RAG_DB.pkl"
questions = [
                "Wie was Napoleon Bonaparte?",
                "Was Napoleon Bonaparte een held of een schurk?"
            ]
languages = [
                "nl",
                "nl"
            ]

sRAG = SimpleRAG(language_model_name=model_name, test=True, dataset_path=dataset_path)
sRAG.set_questions(questions, languages)
answers = sRAG.generate_response(top_k=top_k, max_new_tokens=max_new_tokens)

for q, a in zip(questions, answers):
    print(f"Question: {q}")
    print(f"Answer: {a}")
    print()
