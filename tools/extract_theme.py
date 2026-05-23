import sys
import argparse
import pretty_midi
import numpy as np
import os


def skyline(instrument, n_notes: int = 20, chord_tolerance: float = 0.05) -> list:
    notes = sorted(instrument.notes, key=lambda n: n.start)
    melody = []
    i = 0

    while i < len(notes) and len(melody) < n_notes:
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and abs(notes[j].start - notes[i].start) < chord_tolerance:
            group.append(notes[j])
            j += 1
        melody.append(max(group, key=lambda n: n.pitch))
        i = j

    return melody

def fold_octaves(intervals: list, threshold: int = 6) -> list:
    folded = []
    for interval in intervals:
        while interval > threshold:
            interval -= 12
        while interval < -threshold:
            interval += 12
        folded.append(interval)
    return folded

def extract(midi_path: str, instrument_index: int = 0, n_notes: int = 20) -> None:
    midi = pretty_midi.PrettyMIDI(midi_path)

    print(f"\nFile:      {midi_path.split('/')[-1].split(chr(92))[-1]}")
    print(f"Duration:  {midi.get_end_time():.1f} seconds")
    print(f"Instruments ({len(midi.instruments)}):")
    for i, inst in enumerate(midi.instruments):
        marker = " ←" if i == instrument_index else ""
        name = inst.name.strip() or "(unnamed)"
        print(f"  [{i}] {name} — program {inst.program}, {len(inst.notes)} notes{marker}")

    inst = midi.instruments[instrument_index]
    melody = skyline(inst, n_notes=n_notes)

    if not melody:
        print("\nNo notes found in this instrument.")
        return

    names = [pretty_midi.note_number_to_name(n.pitch) for n in melody]
    print(f"\nSkyline melody — instrument [{instrument_index}], first {len(melody)} notes:")
    print("  " + "  ".join(f"{n:<4}" for n in names))

    pitches = np.array([n.pitch for n in melody])
    intervals = fold_octaves([int(x) for x in np.diff(pitches)])
    print(f"\nIntervals ({len(intervals)}):")
    print(f"  {intervals}")

    print(f"\n--- Paste into TEST_CORPUS ---")
    print(f'{{')
    print(f'    "title": "???",')
    print(f'    "composer": "???",')
    print(f'    "movement": None,')
    print(f'    "intervals": np.array({intervals})')
    print(f'}},')
    print(f'------------------------------')

def batch_extract(midi_dir: str, instrument_index: int = 0, n_notes: int = 20) -> None:
    """Run extract on every MIDI file in a directory."""
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    
    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return
    
    print(f"Found {len(midi_files)} MIDI files in {midi_dir}\n")
    print("=" * 60)
    
    for filename in sorted(midi_files):
        path = os.path.join(midi_dir, filename)
        try:
            extract(path, instrument_index, n_notes)
        except Exception as e:
            print(f"\nFailed on {filename}: {e}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Extract theme intervals from a MIDI file.")
    parser.add_argument("midi_path", help="Path to the MIDI file")
    parser.add_argument("--instrument", type=int, default=0)
    parser.add_argument("--notes", type=int, default=20)
    parser.add_argument("--batch", action="store_true",
                        help="Process all MIDI files in a directory")
    args = parser.parse_args()
    if args.batch or os.path.isdir(args.midi_path):
        batch_extract(args.midi_path, args.instrument, args.notes)
    else:
        extract(args.midi_path, args.instrument, args.notes)


if __name__ == "__main__":
    main()