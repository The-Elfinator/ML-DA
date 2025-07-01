import tkinter as tk
from tkinter import messagebox
import torch
from transformers import AutoTokenizer
from transformer import TransformerRussianToEnglish, generate_square_subsequent_mask

text_tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en', use_fast=True)
vocab_size = text_tokenizer.vocab_size
model_dim = 512

model = TransformerRussianToEnglish(vocab_size=vocab_size, model_dim=model_dim)
model.load_state_dict(torch.load("transformer_weights_10_epochs.pth", map_location=torch.device('mps')))
model.eval()

def translate_text():
    input_text = input_field.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showerror("Ошибка", "Введите текст для перевода.")
        return

    src_ids = text_tokenizer.encode(input_text, return_tensors="pt").transpose(0, 1)

    start_token = text_tokenizer.bos_token_id if text_tokenizer.bos_token_id is not None else text_tokenizer.eos_token_id
    if start_token is None:
        start_token = text_tokenizer.pad_token_id

    generated_tokens = [start_token]

    max_length = 50
    with torch.no_grad():
        for i in range(max_length):
            tgt_input_ids = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(1)
            output = model(src_ids, tgt_input_ids, src_mask=None, tgt_mask=None)
            next_token = output[-1, 0].argmax().item()
            generated_tokens.append(next_token)
            if text_tokenizer.eos_token_id is not None and next_token == text_tokenizer.eos_token_id:
                break

    translated_text = text_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    output_field.delete("1.0", tk.END)
    output_field.insert(tk.END, translated_text)
    print(translated_text)

root = tk.Tk()
root.title("Translator")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

input_label = tk.Label(frame, text="Введите текст на русском:")
input_label.pack(anchor="w")

input_field = tk.Text(frame, height=5, width=50)
input_field.pack(pady=5)

translate_button = tk.Button(frame, text="Перевести", command=translate_text)
translate_button.pack(pady=5)

output_label = tk.Label(frame, text="Перевод на английский:")
output_label.pack(anchor="w")

output_field = tk.Text(frame, height=5, width=50, state="normal")
output_field.pack(pady=5)

root.mainloop()
