import sys
import argparse
import pretty_midi
import numpy as np


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
    intervals = list(np.diff(pitches).astype(int))
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


def main():
    parser = argparse.ArgumentParser(description="Extract theme intervals from a MIDI file.")
    parser.add_argument("midi_path", help="Path to the MIDI file")
    parser.add_argument("--instrument", type=int, default=0)
    parser.add_argument("--notes", type=int, default=20)
    args = parser.parse_args()
    extract(args.midi_path, args.instrument, args.notes)


if __name__ == "__main__":
    main()