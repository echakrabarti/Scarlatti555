import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THEME_SCHEMA = {
    "title": str,
    "composer": str,
    "intervals": np.ndarray
}

# hardcoded test corpus
TEST_CORPUS = [
    {
        "title": "Symphony No. 5 in C minor",
        "composer": "Beethoven",
        "movement": 1,
        "intervals": np.array([0, 0, -4])
    },
    {
        "title": "Fur Elise",
        "composer": "Beethoven",
        "movement": None,
        "intervals": np.array([-1, 1, -1, 1, -1, 2, -3, 2])
    },
    {
        "title": "Ode to Joy",
        "composer": "Beethoven",
        "movement": 4,
        "intervals": np.array([0, 0, 1, 2, 2, 1, 0, -1])
    },
    {
        "title": "Eine Kleine Nachtmusik",
        "composer": "Mozart",
        "movement": 1,
        "intervals": np.array([7, -5, 5, -4, 4, -2, 2, -7])
    },
    {
        "title": "The Four Seasons - Spring",
        "composer": "Vivaldi",
        "movement": 1,
        "intervals": np.array([2, 1, 2, 2, 1, 2, 2])
    }
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0]

def sliding_window_match(query: np.ndarray, theme: np.ndarray) -> float:
    query_len = len(query)
    theme_len = len(theme)

    if query_len >= theme_len:
        return cosine_sim(query[:theme_len], theme)
    best_score = -1.0
    for i in range(theme_len - query_len + 1):
        window = theme[i: i+query_len]
        score = cosine_sim(query, window)
        if score > best_score:
            best_score = score
    return best_score

def match(query_intervals: np.ndarray, corpus: list = None, top_n: int = 3)->list: 
    if corpus is None:
        corpus = TEST_CORPUS
    results = []
    for theme in corpus:
        score = sliding_window_match(query_intervals, theme["intervals"])
        results.append({
            "title": theme["title"],
            "composer": theme["composer"],
            "score": float(score)
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]   

def test_match():
    print("running match test...")
    test_query = np.array([0,0,-4])
    results = match(test_query)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['composer']} — {result['title']}: {result['score']:.3f}")
    assert results[0]["title"] == "Symphony No. 5 in C minor", "top match should be beethoven 5th"
    assert results[0]["score"] > 0.99, "exact match should score near 1.0"
    print("\ntest passed.")

if __name__ == "__main__":
    test_match()