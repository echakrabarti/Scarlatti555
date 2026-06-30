"""
match.py — melody matching via trained siamese encoder.
Replaces DTW-based matching. Corpus embeddings are pre-computed 
ONCE at module load time; each query is a fast vector comparison.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np

from app.melody_encoder import MelodyEncoder

QUERY_LENGTH = 15
CORPUS_PATH = "data/lieder_voice_extraction_results.json"
WEIGHTS_PATH = "data/melody_encoder.pt"


def _load_encoder():
    encoder = MelodyEncoder()
    encoder.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    encoder.eval()
    return encoder


def _embed_sequence(encoder, intervals):
    x = torch.tensor(intervals, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([len(intervals)])
    with torch.no_grad():
        embedding = encoder(x, lengths)
    return embedding.squeeze(0)


def _split_title(full_title):
    """
    Titles are formatted like:
    'Schubert, Franz - Winterreise, D.911 - 1 Gute Nacht'
    First segment is the composer, rest is the piece title.
    """
    parts = full_title.split(" - ", 1)
    if len(parts) == 2:
        composer, piece_title = parts
    else:
        composer, piece_title = "Unknown", full_title
    return composer.strip(), piece_title.strip()


def _load_corpus_embeddings(encoder):
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    titles, composers, embeddings = [], [], []
    for piece in corpus:
        if piece["interval_count"] < QUERY_LENGTH:
            continue
        opening = np.array(piece["intervals"][:QUERY_LENGTH])
        emb = _embed_sequence(encoder, opening)

        composer, piece_title = _split_title(piece["title"])
        titles.append(piece_title)
        composers.append(composer)
        embeddings.append(emb)

    matrix = torch.stack(embeddings)
    return titles, composers, matrix

# --- loaded ONCE at import time, not per request ---
print("Loading melody encoder and corpus embeddings...")
ENCODER = _load_encoder()
CORPUS_TITLES, CORPUS_COMPOSERS, CORPUS_MATRIX = _load_corpus_embeddings(ENCODER)
print(f"Loaded {len(CORPUS_TITLES)} corpus embeddings.")


def match(query_intervals: np.ndarray, top_n: int = 3) -> list:
    """
    query_intervals: semitone interval sequence from a hummed query
    (from app/pitch.py). Uses the opening QUERY_LENGTH intervals,
    matching how the corpus and training data were constructed.
    """
    query = np.array(query_intervals[:QUERY_LENGTH])
    if len(query) < 5:
        return []

    query_emb = _embed_sequence(ENCODER, query)
    similarities = F.cosine_similarity(query_emb.unsqueeze(0), CORPUS_MATRIX)
    sorted_sims, sorted_idx = torch.sort(similarities, descending=True)

    results = []
    for i in range(min(top_n, len(sorted_idx))):
        idx = sorted_idx[i].item()
        results.append({
            "title": CORPUS_TITLES[idx],
            "composer": CORPUS_COMPOSERS[idx],
            "movement": None,
            "score": float(sorted_sims[i].item())
        })
    return results