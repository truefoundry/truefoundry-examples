from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Model:
    def __init__(self, model_fqn: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_fqn)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_fqn)

    def infer(self, input_text: str) -> str:
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
