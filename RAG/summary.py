from transformers import AutoTokenizer, AutoModelForCausalLM


class SummaryModel:
    def __init__(self, name):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name)
        self.messages = [
            # {"role": "user", "content": "Who are you?"},
        ]
        self.inputs = {}
        self.outputs = []
        self.decoded_outputs = []

    def set_messages(self, messages):
        self.messages = messages

    def set_input(self):
        self.inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    def generate_outputs(self, max_new_tokens):
        self.outputs = self.model.generate(**self.inputs, max_new_tokens=max_new_tokens)
        self.decoded_outputs = self.tokenizer.decode(self.outputs[0][self.inputs["input_ids"].shape[-1]:])

    # todo: which metric to use?