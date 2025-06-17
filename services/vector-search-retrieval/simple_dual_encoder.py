import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import glob
import os
import json
import torch.optim as optim

# Load the tokenizer (vocab)
with open('cbow/tkn_words_to_ids.pkl', 'rb') as f:
    words_to_ids = pickle.load(f)
vocab_size = len(words_to_ids)
embedding_dim = 128  # Use the dimension you trained with

# Find the latest CBOW checkpoint
checkpoint_files = glob.glob('cbow/checkpoints/*.pth')
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
state_dict = torch.load(latest_checkpoint)

# Create the embedding layer and load weights
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer.weight.data.copy_(state_dict['emb.weight'])
embedding_layer.weight.requires_grad = False  # freeze weights

def average_embedding(token_ids, embedding_layer):
    if not token_ids:  # skip empty
        return None
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
    vectors = embedding_layer(tokens_tensor)
    avg_vector = vectors.mean(dim=0)
    return avg_vector

def triplet_loss(qry, pos, neg, margin=0.2):
    dst_pos = F.cosine_similarity(qry.unsqueeze(0), pos.unsqueeze(0))
    dst_neg = F.cosine_similarity(qry.unsqueeze(0), neg.unsqueeze(0))
    loss = torch.clamp(margin - (dst_pos - dst_neg), min=0.0)
    return loss, dst_pos.item(), dst_neg.item()

# Load real tokenized triples (small subset for speed)
with open('tokenized_triples.json', 'r') as f:
    triples_data = json.load(f)

print("\nRunning on first 5 real triples from the train split:\n")
for i, triple in enumerate(triples_data['train'][:5]):
    qry_tokens = triple['query_tokens']
    pos_tokens = triple['positive_document_tokens']
    neg_tokens = triple['negative_document_tokens']
    qry_text = triple['query']
    pos_text = triple['positive_document']
    neg_text = triple['negative_document']

    qry_vec = average_embedding(qry_tokens, embedding_layer)
    pos_vec = average_embedding(pos_tokens, embedding_layer)
    neg_vec = average_embedding(neg_tokens, embedding_layer)

    if qry_vec is not None and pos_vec is not None and neg_vec is not None:
        loss, sim_pos, sim_neg = triplet_loss(qry_vec, pos_vec, neg_vec)
        print(f"Example {i+1}:")
        print(f"Query: {qry_text}")
        print(f"Positive doc: {pos_text[:100]}...")
        print(f"Negative doc: {neg_text[:100]}...")
        print(f"Cosine similarity (pos): {sim_pos:.4f}")
        print(f"Cosine similarity (neg): {sim_neg:.4f}")
        print(f"Triplet loss: {loss.item():.4f}\n")
    else:
        print(f"Example {i+1}: One of the inputs was empty, skipping this triple.\n")

# Only optimize the RNNs, not the embedding layer
params = list(qryTower.rnn.parameters()) + list(docTower.rnn.parameters())
optimizer = optim.Adam(params, lr=0.001)

num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for triple in triples_data['train']:  # Use all triples
        qry_tokens = triple['query_tokens']
        pos_tokens = triple['positive_document_tokens']
        neg_tokens = triple['negative_document_tokens']

        qry = qryTower(qry_tokens)
        pos = docTower(pos_tokens)
        neg = docTower(neg_tokens)

        if qry is not None and pos is not None and neg is not None:
            dst_pos = F.cosine_similarity(qry.unsqueeze(0), pos.unsqueeze(0))
            dst_neg = F.cosine_similarity(qry.unsqueeze(0), neg.unsqueeze(0))
            dst_mrg = torch.tensor(0.2)
            loss = torch.max(torch.tensor(0.0), dst_mrg - (dst_pos - dst_neg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

    print(f"Epoch {epoch+1}, Avg Loss: {total_loss / max(count,1):.4f}") 