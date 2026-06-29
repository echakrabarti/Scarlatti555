"""
build_voice_piano_pairs.py — walk the entire OpenScore Lieder corpus,
identify voice vs piano parts by name, and extract (voice, full_collapsed)
training pairs.
"""

import os
import json
from music21 import converter

CORPUS_ROOT = "data/openscore_lieder/scores"

# known voice part name variants across languages/conventions
VOICE_NAME_KEYWORDS = [
    "singstimme", "voice", "voce", "gesang", "soprano", 
    "tenor", "alto", "bass", "mezzo", "baritone", "vocal"
]

def is_voice_part(part_name):
    if not part_name:
        return False
    name_lower = part_name.lower()
    return any(keyword in name_lower for keyword in VOICE_NAME_KEYWORDS)

def find_mxl_files(root):
    """Walk the corpus directory tree and find all .mxl files."""
    mxl_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.mxl'):
                mxl_files.append(os.path.join(dirpath, f))
    return mxl_files

def extract_pair(mxl_path):
    """
    Returns (voice_notes, piano_parts) or None if voice part 
    can't be confidently identified.
    """
    try:
        score = converter.parse(mxl_path)
    except Exception as e:
        return None, f"parse_error: {e}"

    voice_part = None
    piano_parts = []

    for part in score.parts:
        if is_voice_part(part.partName):
            voice_part = part
        else:
            piano_parts.append(part)

    if voice_part is None:
        return None, "no_voice_part_found"

    voice_notes = voice_part.flatten().notes
    if len(voice_notes) < 10:
        return None, "voice_part_too_short"

    return {
        "voice_part_name": voice_part.partName,
        "voice_note_count": len(voice_notes),
        "piano_part_count": len(piano_parts),
        "piano_part_names": [p.partName for p in piano_parts],
    }, None


if __name__ == "__main__":
    mxl_files = find_mxl_files(CORPUS_ROOT)
    print(f"Found {len(mxl_files)} .mxl files\n")

    results = []
    errors = []

    for i, path in enumerate(mxl_files):
        if i % 100 == 0:
            print(f"Processing {i}/{len(mxl_files)}...")

        info, error = extract_pair(path)
        if error:
            errors.append({"path": path, "error": error})
        else:
            info["path"] = path
            results.append(info)

    print(f"\n{'='*60}")
    print(f"Successfully processed: {len(results)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nError breakdown:")
        from collections import Counter
        error_types = Counter(e["error"].split(":")[0] for e in errors)
        for err_type, count in error_types.most_common():
            print(f"  {err_type}: {count}")

    # save results for inspection
    os.makedirs("data", exist_ok=True)
    with open("data/lieder_voice_extraction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("data/lieder_voice_extraction_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print(f"\nSaved results to data/lieder_voice_extraction_results.json")
    print(f"Saved errors to data/lieder_voice_extraction_errors.json")