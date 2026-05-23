import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
THEME_SCHEMA = {
    "title": str,
    "composer": str,
    "intervals": np.ndarray
}

# hardcoded test corpus
# TODO: each piece should have multiple themes, if applicable, as NumPy arr
# TODO: ML-built data foundry for themes
TEST_CORPUS = [
    {
        "title": "Symphony No. 5 in C minor",
        "composer": "Beethoven",
        "movement": 1,
        "intervals": np.array([0, 0, -4, 2, 0, 0, -3])
    },
    {
        "title": "Ode to Joy",
        "composer": "Beethoven",
        "movement": 4,
        "intervals": np.array([0, 0, 1, 2, 2, 1, 0, -1])
    },
    {
        "title": "Fur Elise",
        "composer": "Beethoven",
        "movement": None,
        "intervals": np.array([-1, 1, -1, 1, -5, 3, -2, -3, -5, 5, 3, 4, 5, 2, 5, 4, -4, 4, 3])
    },
    {
        "title": "Eine Kleine Nachtmusik",
        "composer": "Mozart",
        "movement": 1,
        "intervals": np.array([-5, 5, -5, 5, -5, 5, 4, 3, -2, -3, 3, -3, 3, -3, -3, 3, 5, 5, 0])
    },
    {
        "title": "The Four Seasons - Spring",
        "composer": "Vivaldi",
        "movement": 1,
        "intervals": np.array([0, 0, 4, 1, 2, 5, 0, 0, -5, -2, -1, -4, 2, -2, -1, 1, 0, 0, 4])
    },
    {
        "title": "Humoresque No. 7",
        "composer": "Dvorak",
        "movement": None,
        "intervals": np.array([2, -2, 2, 2, 3, 2, -2, 2, 3, -1, 3, -2, -1, 3, -2, -3, -5, 3, 0])
    },
    {
        "title": "Nocturne Op. 9 No. 2",
        "composer": "Chopin",
        "movement": None,
        "intervals": np.array([-3, -2, 2, -2, -2, -5, -3, 5, 1, -1, -1, 1, 0, -5, 3, -2, -1, -2, 2])
    },
    {
        "title": "Clair de Lune",
        "composer": "Debussy",
        "movement": None,
        "intervals": np.array([0, -3, 4, 6, 2, -2, 5, 5, 2, -2, 4, -4, 5, 6, -2, 3, -1, 1, 2])
    }
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0]

def dtw_match(query: np.ndarray, theme: np.ndarray) -> float:
    query_2d = query.reshape(-1, 1)
    theme_2d = theme.reshape(-1, 1)
    distance, alignment_path = fastdtw(query_2d, theme_2d, dist=euclidean)
    return 1.0/(1.0+(distance/len(alignment_path)))

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
        score = dtw_match(query_intervals, theme["intervals"])
        results.append({
            "title": theme["title"],
            "composer": theme["composer"],
            "score": float(score)
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]   

def test_match():
    print("running match test...")
    test_query = np.array([0, 0, -4, 2, 0, 0, -3])
    results = match(test_query)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['composer']} — {result['title']}: {result['score']:.3f}")
    assert results[0]["title"] == "Symphony No. 5 in C minor", "top match should be beethoven 5th"
    assert results[0]["score"] > 0.99, "exact match should score near 1.0"
    print("\ntest passed.")

if __name__ == "__main__":
    test_match()