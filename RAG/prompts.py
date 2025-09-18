class Prompt:
    def __init__(self, language, question, context):
        self.language = language
        self.context = context
        self.question = question
        self.chat_prompt = []

        if self.language == 'nl':
            self.set_nl_chat(self.context, self.question)
        if self.language == 'en':
            self.set_en_chat(self.context, self.question)

    def set_nl_chat(self, context: str, question: str):
        system_prompt = f'''Geef een uitgebreid antwoord op de vraag, waarbij je je kennis en de informatie in de juiste context plaatst.
Reageer alleen op de gestelde vraag; je antwoord moet beknopt en relevant zijn voor de vraag. Reageer in het Nederlands.'''

        user_prompt = f'''
        Context:
        {context}
        ---
        Dit is de vraag die je moet beantwoorden.
        
        Vraag: {question}
        '''
        self.chat_prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def set_en_chat(self, context: str, question: str):
        system_prompt = f'''Provide a comprehensive answer to the question, using your knowledge and the information in the given context.
Respond only to the given question; your answer should be concise and relevant to the question. Respond in English.'''

        user_prompt = f'''
            Context:
            {context}
            ---
            This is the question you must answer.

            Question: {question}
            '''
        self.chat_prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def set_fr_chat(self, context: str, question:str):
        """ TODO: same as above but in french """
        pass
