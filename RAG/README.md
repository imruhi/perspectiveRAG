# RAG experimental set-up

This branch contains the RAG setup:
* `prompts.py` **contains the structure of the chat prompts per language**


* `questions.py` **contains the questions used in QA divided per language** 


* `evaluation` contains the code needed to evaluate model answers, relating to frames, UD profiling, topic modeling and similarity metrics


* `advanced_rag.py` contains the code to run retrieval and QA 


* `main.py` contains the script to run the experiment and generate results


*  `parmas.json` contains the (hyper) parameters which were set relating to languages, models, encoders and vector storage
