"""
self_match_eval.py — test whether AUGMENTED, truncated melody excerpts 
correctly retrieve their own source piece from the full Lieder corpus.

Augmentation simulates realistic humming imperfections: pitch jitter,
note dropout, occasional octave errors. This tests matching robustness
under humming-like noise, NOT a claim that real humans will match at
this exact rate -- that still requires validation against real human
hums (see Step 2 in project notes).
"""

import json
import random
import numpy as np
import time
from oauthlib.uri_validate import query
from app.match import sliding_window_dtw

QUERY_LENGTH = 15
N_TRIALS = 1  # start small, scale up after timing test

# augmentation parameters -- tuned to be plausible, not validated
PITCH_JITTER_STD = 0.15      # semitones, gaussian noise per interval
NOTE_DROPOUT_PROB = 0.08     # probability of dropping any given note
OCTAVE_ERROR_PROB = 0.03     # probability of an accidental octave jump

start_time = time.perf_counter()

def augment_query(intervals, rng):
    """
    Apply humming-like noise to a clean interval sequence.
    Returns a new, distorted interval list -- NOT a claim that 
    this matches real human humming exactly, just a plausible 
    stress test for the matcher.
    """
    intervals = list(intervals)

    # note dropout -- randomly remove some notes (merges adjacent intervals)
    if len(intervals) > 5:
        keep_mask = rng.random(len(intervals)) > NOTE_DROPOUT_PROB
        intervals = [iv for iv, keep in zip(intervals, keep_mask) if keep]

    if len(intervals) < 5:
        return None  # dropped too much, discard this trial

    # pitch jitter -- gaussian noise on each interval
    jitter = rng.normal(0, PITCH_JITTER_STD, len(intervals))
    intervals = [iv + j for iv, j in zip(intervals, jitter)]

    # occasional octave error -- shift one random interval by +/- 12
    if rng.random() < OCTAVE_ERROR_PROB:
        idx = rng.integers(0, len(intervals))
        intervals[idx] += rng.choice([-12, 12])

    return np.array(intervals)


def evaluate_self_matching(corpus, n_trials=N_TRIALS, augment=True):
    rng = np.random.default_rng(seed=42)  # reproducible

    correct = 0
    total = 0
    results = []

    candidates = [p for p in corpus if p["interval_count"] >= QUERY_LENGTH + 5]
    test_pieces = random.sample(candidates, min(n_trials, len(candidates)))

    for i, piece in enumerate(test_pieces):
        if i % 20 == 0:
            print(f"Evaluating {i}/{len(test_pieces)}...")

        full_intervals = np.array(piece["intervals"])
        start = random.randint(0, len(full_intervals) - QUERY_LENGTH)
        clean_query = full_intervals[start:start + QUERY_LENGTH]

        if augment:
            query = augment_query(clean_query, rng)
            if query is None:
                continue  # skip if dropout removed too much
        else:
            query = clean_query

        scores = []
        for candidate in corpus:
            cand_intervals = np.array(candidate["intervals"])
            if len(cand_intervals) < 3:
                continue
            score = sliding_window_dtw(query, cand_intervals)
            scores.append((candidate["title"], score))
            

        scores.sort(key=lambda x: x[1], reverse=True)
        # temporarily add inside the trial loop, right after computing scores:
        print(f"\n--- DEBUG ---")
        print(f"True title: {piece['title']}")
        print(f"Query (first 10): {query[:10]}")
        print(f"Top 5 scores:")
        for title, score in scores[:5]:
            print(f"  {title}: {score}")
        print(f"--- END DEBUG ---\n")
        top_match, top_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0

        is_correct = (top_match == piece["title"])
        correct += int(is_correct)
        total += 1

        results.append({
            "true_title": piece["title"],
            "predicted": top_match,
            "correct": is_correct,
            "top_score": round(top_score, 4),
            "gap": round(top_score - second_score, 4),
            "augmented": augment
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Accuracy ({'augmented' if augment else 'clean'}): {correct}/{total} = {accuracy:.1%}")
    return results, accuracy


if __name__ == "__main__":
    with open("data/lieder_voice_extraction_results.json") as f:
        corpus = json.load(f)

    print(f"Corpus size: {len(corpus)} pieces\n")

    print("=== CLEAN (no augmentation) ===")
    clean_results, clean_acc = evaluate_self_matching(corpus, augment=False)

    print("\n=== AUGMENTED (simulated humming noise) ===")
    aug_results, aug_acc = evaluate_self_matching(corpus, augment=True)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Script execution time: {execution_time:.4f} seconds")
    print(f"  Clean accuracy:     {clean_acc:.1%}")
    print(f"  Augmented accuracy: {aug_acc:.1%}")
    print(f"  Degradation:        {clean_acc - aug_acc:.1%}")

    with open("data/self_match_eval_results.json", "w") as f:
        json.dump({"clean": clean_results, "augmented": aug_results}, f, indent=2)