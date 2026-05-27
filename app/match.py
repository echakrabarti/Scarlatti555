import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


TEST_CORPUS = [
    {
        "title": "Fur Elise",
        "composer": "Beethoven",
        "movement": None,
        "full_intervals": [-1, 1, 0, -1, 1, -5, 0, 3, -2, -3, 0, -5, 5, 3, 0, 4, 5, 2, 0, 5, 4, -4, 4, 3, 1, -1, 1, 0, -1, 1, -5, 0, 3, -2, -3, 0, -5, 5, 3, 0]
    },
    {
        "title": "Humoresque No. 7",
        "composer": "Dvorak",
        "movement": None,
        "full_intervals": [2, 2, 5, 3, 0, 2, -2, 2, -5, 0, 0, 3, -3, -5, -2, -2, 2, 2, 5, 0, 3, -3, 2, -2, 2, 2, 3, -3, -5, 0, 2, -2, 2, 5, 3, 0, 2, -2, 2, -5]
    },
    {
        "title": "Nocturne Op. 9 No. 2",
        "composer": "Chopin",
        "movement": None,
        "full_intervals": [-3, -2, 2, -2, -2, -5, -3, -6, -1, 0, -5, 3, -2, -1, -2, 2, -5, 1, -3, -2, 2, -2, -2, -5, -3, 5, 1, -1, -1, 1, 0, -5, 3, -2, -1, -2, 2, -5, 1, -3]
    },
    {
        "title": "Clair de Lune",
        "composer": "Debussy",
        "movement": None,
        "full_intervals": [0, -3, 0, 4, 6, 2, -2, 0, 5, -5, 0, -2, 4, 0, -4, 5, -5, -1, 1, -1, 2, -2, 0, 3, -3, 0, 4, 6, 2, -2, 0, 5, -5, 0, -2, 4, 0, -4, 5, -5]
    },
    {
        "title": "Eine Kleine Nachtmusik",
        "composer": "Mozart",
        "movement": 1,
        "full_intervals": [-5, 5, -5, 5, -5, 5, 4, 3, -2, -3, 3, -3, 3, -3, -3, 3, 5, 5, 0, 4, -4, 0, 0, -5, 5, -5, 5, -5, 5, 4, 3, -2, -3, 3, -3, 3, -3, -3, 3, 5]
    },
    {
        "title": "The Four Seasons - Spring",
        "composer": "Vivaldi",
        "movement": 1,
        "full_intervals": [0, 0, 4, 1, 2, 5, 0, 0, -5, -2, -1, -4, 2, -2, -1, 0, 1, 0, 0, 4, 1, 2, 5, 0, 0, -5, -2, -1, -4, 2, -2, -1, 0, 1, 0, 0, 4, 1, 2, 5]
    }
]


def dtw_match(query: np.ndarray, theme: np.ndarray) -> float:
    query_2d = query.reshape(-1, 1)
    theme_2d = theme.reshape(-1, 1)
    distance, path = fastdtw(query_2d, theme_2d, dist=euclidean)
    normalized_distance = distance / len(path)
    return 1.0 / (1.0 + normalized_distance)


def sliding_window_dtw(query: np.ndarray, full_theme: np.ndarray) -> float:
    query_len = len(query)
    theme_len = len(full_theme)
    window_size = min(int(query_len * 1.5), theme_len)

    if theme_len <= window_size:
        return dtw_match(query, full_theme)

    best_score = 0.0
    for start in range(0, theme_len - window_size + 1, 1):
        window = full_theme[start:start + window_size]
        score = dtw_match(query, window)
        if score > best_score:
            best_score = score

    return best_score


def match(query_intervals: np.ndarray,
          corpus: list = None,
          top_n: int = 3) -> list:
    if corpus is None:
        corpus = TEST_CORPUS

    results = []
    for theme in corpus:
        full = np.array(theme["full_intervals"])
        score = sliding_window_dtw(query_intervals, full)
        results.append({
            "title": theme["title"],
            "composer": theme["composer"],
            "movement": theme.get("movement"),
            "score": float(score)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def test_match():
    print("running sliding window DTW test...")

    test_query = np.array([-1, 1, 0, -1, 1, -5, 0, 3])
    results = match(test_query)

    print(f"\ntop {len(results)} matches:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['composer']} — {r['title']}: {r['score']:.4f}")

    assert results[0]["title"] == "Fur Elise", \
        f"Expected Fur Elise, got {results[0]['title']}"
    assert results[0]["score"] > 0.5, \
        f"Expected score > 0.5, got {results[0]['score']}"

    print("\ntest passed.")


if __name__ == "__main__":
    test_match()