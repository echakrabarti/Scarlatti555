"""
eval_siamese.py — evaluate the trained melody encoder across three 
noise tiers: clean (no distortion), high_accuracy (trained singer, 
very tight pitch control), low_accuracy (untrained hummer, the 
original noise model). Reports top-1, top-3 accuracy, and mean rank 
of the true piece for each tier.
"""

import itertools
import json
import torch
import torch.nn.functional as F
import numpy as np

from tools.train_siamese import MelodyEncoder

QUERY_LENGTH = 15
N_TRIALS = 20

NOISE_PROFILES = {
    "clean": None,  # no augmentation at all

    "high_accuracy": {
        "jitter_range": (0.01, 0.03),   # trained singer, very tight pitch control
        "dropout_prob": 0.01,
        "octave_error_prob": 0.0,
    },

    "low_accuracy": {
        "jitter_range": (0.1, 0.5),     # untrained hummer
        "dropout_prob": 0.08,
        "octave_error_prob": 0.03,
    },
}


def augment_query(intervals, rng, profile_name):
    if profile_name == "clean":
        return np.array(intervals)

    profile = NOISE_PROFILES[profile_name]
    intervals = list(intervals)

    if len(intervals) > 5:
        keep_mask = rng.random(len(intervals)) > profile["dropout_prob"]
        intervals = [iv for iv, keep in zip(intervals, keep_mask) if keep]
    if len(intervals) < 5:
        return None

    jitter_std = rng.uniform(*profile["jitter_range"])
    jitter = rng.normal(0, jitter_std, len(intervals))
    intervals = [iv + j for iv, j in zip(intervals, jitter)]

    if rng.random() < profile["octave_error_prob"]:
        idx = rng.integers(0, len(intervals))
        intervals[idx] += rng.choice([-12, 12])

    return np.array(intervals)


def embed_sequence(encoder, intervals):
    x = torch.tensor(intervals, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([len(intervals)])
    with torch.no_grad():
        embedding = encoder(x, lengths)
    return embedding.squeeze(0)


def evaluate(corpus, encoder, n_trials=N_TRIALS, profile_name="clean", seed=42):
    rng = np.random.default_rng(seed)
    eligible = [p for p in corpus if p["interval_count"] >= QUERY_LENGTH + 5]

    print("Pre-computing corpus embeddings (opening 15 intervals only)...")
    corpus_embeddings = []
    for i, piece in enumerate(eligible):
        if i % 200 == 0:
            print(f"  {i}/{len(eligible)}...")
        opening = np.array(piece["intervals"][:QUERY_LENGTH])
        emb = embed_sequence(encoder, opening)
        corpus_embeddings.append((piece["title"], emb))

    corpus_matrix = torch.stack([e for _, e in corpus_embeddings])
    corpus_titles = [t for t, _ in corpus_embeddings]
    print(f"  Done. {len(corpus_embeddings)} corpus embeddings (1 per piece).")

    sample_idx = np.random.choice(len(corpus_embeddings), size=min(100, len(corpus_embeddings)), replace=False)
    sims = []
    for i, j in itertools.combinations(sample_idx, 2):
        sim = F.cosine_similarity(
            corpus_matrix[i].unsqueeze(0),
            corpus_matrix[j].unsqueeze(0)
        )
        sims.append(sim.item())
    print(f"\nEmbedding space stats (random pairs):")
    print(f"  Mean: {np.mean(sims):.4f}, Std: {np.std(sims):.4f}, "
          f"Min: {np.min(sims):.4f}, Max: {np.max(sims):.4f}")
    print(f"  (Healthy space: mean near 0, std > 0.3, range spans negative to positive)")

    test_pieces = rng.choice(eligible, size=min(n_trials, len(eligible)), replace=False)

    correct = 0
    results = []

    for piece in test_pieces:
        full_intervals = np.array(piece["intervals"])
        clean_excerpt = full_intervals[:QUERY_LENGTH]

        query = augment_query(clean_excerpt, rng, profile_name)
        if query is None:
            continue

        query_emb = embed_sequence(encoder, query)
        similarities = F.cosine_similarity(query_emb.unsqueeze(0), corpus_matrix)
        sorted_sims, sorted_idx = torch.sort(similarities, descending=True)

        top3_titles = [corpus_titles[i] for i in sorted_idx[:3].tolist()]
        top3_correct = piece["title"] in top3_titles

        true_idx = corpus_titles.index(piece["title"])
        true_score = similarities[true_idx].item()
        true_rank = (similarities > true_score).sum().item() + 1

        top_idx = torch.argmax(similarities).item()
        top_title = corpus_titles[top_idx]
        top_score = similarities[top_idx].item()
        second_score = sorted_sims[1].item() if len(sorted_sims) > 1 else 0

        is_correct = (top_title == piece["title"])
        correct += int(is_correct)

        print(f"\n  True: {piece['title'][:60]}")
        print(f"  True piece rank: {true_rank}/{len(eligible)}, score: {true_score:.4f}")
        print(f"  Predicted: {top_title[:60]}")
        print(f"  Score: {top_score:.4f}, Gap: {top_score - second_score:.4f}, "
              f"Correct: {'✓' if is_correct else '✗'}, Top3: {'✓' if top3_correct else '✗'}")

        results.append({
            "true_title": piece["title"],
            "predicted": top_title,
            "correct": is_correct,
            "top3_correct": top3_correct,
            "true_rank": true_rank,
            "true_score": round(true_score, 4),
            "top_score": round(top_score, 4),
            "gap": round(top_score - second_score, 4)
        })

    accuracy = correct / len(results) if results else 0
    top3_accuracy = sum(r["top3_correct"] for r in results) / len(results) if results else 0
    avg_rank = np.mean([r["true_rank"] for r in results]) if results else 0

    print(f"\nAccuracy ({profile_name}): {correct}/{len(results)} = {accuracy:.1%}")
    print(f"Top-3 accuracy ({profile_name}): {top3_accuracy:.1%}")
    print(f"Mean true-piece rank ({profile_name}): {avg_rank:.1f} / {len(eligible)} "
          f"(random baseline would be ~{len(eligible)//2})")

    return results, accuracy, top3_accuracy


if __name__ == "__main__":
    with open("data/lieder_voice_extraction_results.json") as f:
        corpus = json.load(f)

    encoder = MelodyEncoder()
    encoder.load_state_dict(torch.load("data/melody_encoder.pt"))
    encoder.eval()

    all_results = {}
    for profile in ["clean", "high_accuracy", "low_accuracy"]:
        print(f"\n{'='*60}")
        print(f"=== {profile.upper()} ===")
        results, acc, top3 = evaluate(corpus, encoder, profile_name=profile)
        all_results[profile] = {"results": results, "accuracy": acc, "top3_accuracy": top3}

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    for profile, data in all_results.items():
        print(f"  {profile:15s}: top-1={data['accuracy']:.1%}, top-3={data['top3_accuracy']:.1%}")

    with open("data/self_match_eval_results.json", "w") as f:
        json.dump({k: v["results"] for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Full results saved to data/self_match_eval_results.json")