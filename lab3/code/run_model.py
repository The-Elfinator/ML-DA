import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from transformers import AutoTokenizer
from transformer import TranslationDataset, TransformerRussianToEnglish, generate_square_subsequent_mask
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import sacrebleu

BATCH_SIZE = 32

data = []
with open('../../lab3/Dataset/rus.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            russian, english = parts[0], parts[1]
            data.append((english, russian))

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en', use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

print(f"Tokenizer pad_token: {tokenizer.pad_token}")
print(f"Tokenizer cls_token: {tokenizer.cls_token}")
print(f"Tokenizer sep_token: {tokenizer.sep_token}")
print(f"Tokenizer bos_token: {tokenizer.bos_token}")
print(f"Tokenizer eos_token: {tokenizer.eos_token}")


def collate_fn(batch):
    src_texts, tgt_texts = zip(*batch)

    src_encodings = tokenizer(
        list(src_texts),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    tgt_encodings = tokenizer(
        list(tgt_texts),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    src_ids = src_encodings['input_ids']
    tgt_ids = tgt_encodings['input_ids']

    return src_ids, tgt_ids


dataset = TranslationDataset(tokenizer, data)
subset_size = int(len(dataset) * 1)
remaining_size = len(dataset) - subset_size
subset, _ = random_split(dataset, [subset_size, remaining_size])
dataset = subset
train_size = int(0.9 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

vocab_size = tokenizer.vocab_size
model_dim = 512

model = TransformerRussianToEnglish(vocab_size=vocab_size, model_dim=model_dim)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется устройство CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Используется устройство MPS")
else:
    device = torch.device("cpu")
    print("Используется устройство CPU")

model.to(device)

pad_idx = tokenizer.pad_token_id
criterion = CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
losses_per_step = []

for epoch in range(num_epochs):
    model.train()
    print(f"\n=== Starting epoch: {epoch + 1} ===")
    print(f"train_loader len: {len(train_loader)}")

    for step, (src, tgt) in enumerate(train_loader):
        src_ids = src.transpose(0, 1).to(device)
        tgt_ids = tgt.transpose(0, 1).to(device)

        tgt_input = tgt_ids[:-1, :]
        tgt_output = tgt_ids[1:, :]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)

        optimizer.zero_grad()
        output = model(src_ids, tgt_input, src_mask=None, tgt_mask=tgt_mask)
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        losses_per_step.append(loss.item())

plt.figure(figsize=(10, 6))
plt.plot(losses_per_step, label='Training Loss per Step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss per Training Step')
plt.legend()
plt.grid(True)
plt.savefig('../lab3/training_loss_per_step_15_epochs.png')
plt.close()

model.eval()

all_hypotheses = []
all_references = []

start_token = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
if start_token is None:
    start_token = tokenizer.pad_token_id
eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
if eos_token_id is None:
    eos_token_id = tokenizer.pad_token_id

max_length = 50
current_idx = 0
print(f'\nTest loader len: {len(test_loader)}')

with torch.no_grad():
    for src, tgt in test_loader:
        src_ids = src.transpose(0, 1).to(device)  # Shape: [src_seq_len, 1]
        print(
            f'\n--- Translating sample {current_idx + 1} --- referense : {len(all_references)} hyp: {len(all_hypotheses)}')
        src_decoded = tokenizer.decode(src_ids[:, 0].cpu().numpy(), skip_special_tokens=True)

        generated_tokens = [start_token]

        for _ in range(max_length):
            tgt_input_ids = torch.tensor(generated_tokens, dtype=torch.long, device=device).unsqueeze(
                1
            )
            output = model(src_ids, tgt_input_ids, src_mask=None, tgt_mask=None)
            next_token = output[-1, 0].argmax().item()
            generated_tokens.append(next_token)
        hypothesis = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        ref_ids = tgt[0].tolist()
        reference = tokenizer.decode(ref_ids, skip_special_tokens=True)
        all_hypotheses.append(hypothesis)
        all_references.append(reference)
        current_idx += 1

torch.save(model.state_dict(), "../transformer_weights_15_epochs.pth")

bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
print(f"\nBLEU score: {bleu.score:.2f}")
torch.save(model.state_dict(), "../transformer_weights_15_epochs_1.pth")

