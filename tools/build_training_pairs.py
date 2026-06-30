import json
import numpy as np
 
QUERY_LENGTH = 15
N_PAIRS = 10000
 
PITCH_JITTER_STD = 0.4
NOTE_DROPOUT_PROB = 0.08
OCTAVE_ERROR_PROB = 0.03
 
def augment_query(intervals, rng):
    intervals = list(intervals)
    
    # randomly pick which tier this training example simulates
    if rng.random() < 0.5:
        jitter_range = (0.01, 0.03)   # high-accuracy singer
        dropout_prob = 0.01
        octave_prob = 0.0
    else:
        jitter_range = (0.1, 0.5)      # low-accuracy hummer
        dropout_prob = 0.08
        octave_prob = 0.03

    if len(intervals) > 5:
        keep_mask = rng.random(len(intervals)) > dropout_prob
        intervals = [iv for iv, keep in zip(intervals, keep_mask) if keep]
    if len(intervals) < 5:
        return None

    jitter_std = rng.uniform(*jitter_range)
    jitter = rng.normal(0, jitter_std, len(intervals))
    intervals = [iv + j for iv, j in zip(intervals, jitter)]

    if rng.random() < octave_prob:
        idx = rng.integers(0, len(intervals))
        intervals[idx] += rng.choice([-12, 12])

    return np.array(intervals) 
 
def extract_excerpt(full_intervals, rng, length=QUERY_LENGTH):
    if len(full_intervals) < length:
        return None
    start = int(rng.integers(0, len(full_intervals) - length + 1))
    return np.array(full_intervals[start:start + length])
 
 
def generate_pairs(corpus, n_pairs=N_PAIRS, seed=42):
    rng = np.random.default_rng(seed)
 
    eligible = [p for p in corpus if p["interval_count"] >= QUERY_LENGTH + 5]
    print(f"{len(eligible)} pieces eligible out of {len(corpus)}")
 
    pairs = []
 
    for i in range(n_pairs):
        if i % 500 == 0:
            print(f"Generating pair {i}/{n_pairs}...")
 
        if i % 2 == 0:
            # POSITIVE pair: same excerpt, clean vs augmented
            piece = eligible[int(rng.integers(0, len(eligible)))]
            full = np.array(piece["intervals"])
            clean_excerpt = extract_excerpt(full, rng)
            if clean_excerpt is None:
                continue
            aug_excerpt = augment_query(clean_excerpt, rng)
            if aug_excerpt is None:
                continue
            pairs.append({
                "a": clean_excerpt.tolist(),
                "b": aug_excerpt.tolist(),
                "label": 1,
                "piece_a": piece["title"],
                "piece_b": piece["title"],
            })
        else:
            # NEGATIVE pair: two different pieces, both clean
            idx_a, idx_b = rng.choice(len(eligible), size=2, replace=False)
            piece_a = eligible[int(idx_a)]
            piece_b = eligible[int(idx_b)]
            full_a = np.array(piece_a["intervals"])
            full_b = np.array(piece_b["intervals"])
            excerpt_a = extract_excerpt(full_a, rng)
            excerpt_b = extract_excerpt(full_b, rng)
            if excerpt_a is None or excerpt_b is None:
                continue
            pairs.append({
                "a": excerpt_a.tolist(),
                "b": excerpt_b.tolist(),
                "label": 0,
                "piece_a": piece_a["title"],
                "piece_b": piece_b["title"],
            })
 
    n_pos = sum(p["label"] for p in pairs)
    n_neg = len(pairs) - n_pos
    print(f"\nGenerated {len(pairs)} pairs ({n_pos} positive, {n_neg} negative)")
    return pairs
 
 
if __name__ == "__main__":
    with open("data/lieder_voice_extraction_results.json") as f:
        corpus = json.load(f)
 
    pairs = generate_pairs(corpus, n_pairs=N_PAIRS)
 
    with open("data/training_pairs.json", "w") as f:
        json.dump(pairs, f)
 
    print("Saved to data/training_pairs.json")