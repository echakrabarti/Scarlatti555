"""
build_voice_piano_pairs.py — walk the entire OpenScore Lieder corpus,
identify voice vs piano parts by name, and extract full interval 
sequences for evaluation and (later) training.
"""

import os
import json
from music21 import converter

CORPUS_ROOT = "data/openscore_lieder/scores"

VOICE_NAME_KEYWORDS = [
    "singstimme", "voice", "voce", "gesang", "soprano", 
    "tenor", "alto", "bass", "mezzo", "baritone", "vocal"
]

OCTAVE_FOLD_THRESHOLD = 6
MIN_NOTE_COUNT = 10  # skip pieces with too few notes to be useful


def is_voice_part(part_name):
    if not part_name:
        return False
    name_lower = part_name.lower()
    return any(keyword in name_lower for keyword in VOICE_NAME_KEYWORDS)


def fold_octaves(intervals, threshold=OCTAVE_FOLD_THRESHOLD):
    folded = []
    for interval in intervals:
        while interval > threshold:
            interval -= 12
        while interval < -threshold:
            interval += 12
        folded.append(interval)
    return folded


def find_mxl_files(root):
    mxl_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.mxl'):
                mxl_files.append(os.path.join(dirpath, f))
    return mxl_files


def extract_pair(mxl_path):
    """
    Returns (info_dict, error) where info_dict now includes the 
    full semitone interval sequence for the voice part, not just 
    summary stats.
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
    if len(voice_notes) < MIN_NOTE_COUNT:
        return None, "voice_part_too_short"

    # extract pitch sequence (MIDI numbers) in order, skipping rests/chords
    # for a monophonic voice line, take the single pitch per note event
    pitches = []
    for n in voice_notes:
        if hasattr(n, 'pitch'):  # single note
            pitches.append(n.pitch.midi)
        elif hasattr(n, 'pitches') and len(n.pitches) > 0:  # chord — rare in voice parts, take top note
            pitches.append(max(p.midi for p in n.pitches))

    if len(pitches) < MIN_NOTE_COUNT:
        return None, "insufficient_pitches_extracted"

    # compute semitone intervals between consecutive notes
    raw_intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
    folded_intervals = fold_octaves(raw_intervals)

    return {
        "voice_part_name": voice_part.partName,
        "voice_note_count": len(pitches),
        "piano_part_count": len(piano_parts),
        "piano_part_names": [p.partName for p in piano_parts],
        "intervals": folded_intervals,        # <-- the new, critical field
        "interval_count": len(folded_intervals),
        "path": mxl_path,
    }, None


def piece_title_from_path(path, root):
    """
    Derive a readable title from the folder structure, e.g.
    'Schubert,_Franz/Winterreise,_D.911/1_Gute_Nacht/file.mxl'
    -> 'Schubert, Franz - Winterreise, D.911 - 1 Gute Nacht'
    """
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)[:-1]  # drop filename
    readable = [p.replace('_', ' ') for p in parts]
    return " - ".join(readable)


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
            info["title"] = piece_title_from_path(path, CORPUS_ROOT)
            results.append(info)

    print(f"\n{'='*60}")
    print(f"Successfully processed: {len(results)}")
    print(f"Errors: {len(errors)}")

    if errors:
        from collections import Counter
        error_types = Counter(e["error"].split(":")[0] for e in errors)
        print(f"\nError breakdown:")
        for err_type, count in error_types.most_common():
            print(f"  {err_type}: {count}")

    os.makedirs("data", exist_ok=True)
    with open("data/lieder_voice_extraction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("data/lieder_voice_extraction_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print(f"\nSaved {len(results)} pieces with full interval sequences")
    print(f"  -> data/lieder_voice_extraction_results.json")