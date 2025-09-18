
class Prompt:
    def __init__(self, language, question, context):
        self.language = language
        self.context = context
        self.question = question
        self.chat_prompt = []

        if self.language == 'nl':
            self.set_nl_chat(self.context, self.question)

    def set_nl_chat(self, context: str, question: str):
        system_prompt = f'''Geef een uitgebreid antwoord op de vraag, waarbij je je kennis en de informatie in de juiste context plaatst.
Reageer alleen op de gestelde vraag; je antwoord moet beknopt en relevant zijn voor de vraag.
Vermeld indien relevant het nummer van het brondocument.'''

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
