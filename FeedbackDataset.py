import torch
from torch.utils.data import Dataset

def format_input(entry):
    instruction_text = (
        f"Dưới đây là một feedback của khách hàng."
        f"Hãy đánh giá mức độ cũng như phân loại của chúng."
        f"\n\n### Input:\n{entry['input']}"
    )

    return instruction_text

class FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output_1'] + entry['output_2'] + entry['output_3']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)