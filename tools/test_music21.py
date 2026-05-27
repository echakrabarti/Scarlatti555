from music21 import converter
from pretty_midi import note_number_to_name
from collections import defaultdict
import numpy as np
import os

MIDI_FILES = [
    ("data/midi/fur_elise.mid",              "Fur Elise",           "Beethoven"),
    ("data/midi/humoresque_7.mid",           "Humoresque No. 7",    "Dvorak"),
    ("data/midi/nocturne_9_2.mid",           "Nocturne Op. 9 No. 2","Chopin"),
    ("data/midi/debussy_clair_de_lune.mid",  "Clair de Lune",       "Debussy"),
    ("data/midi/einekleine.mid",             "Eine Kleine",         "Mozart"),
    ("data/midi/vivaldi_spring.mid",         "Spring",              "Vivaldi"),
]

MIN_DURATION = 0.25

def fold_octaves(intervals, threshold=6):
    folded = []
    for interval in intervals:
        while interval > threshold:
            interval -= 12
        while interval < -threshold:
            interval += 12
        folded.append(interval)
    return folded

def extract_melody(midi_path, instrument_index=0):
    score = converter.parse(midi_path)
    part = score.parts[instrument_index]
    all_notes = part.flatten().notes

    by_offset = defaultdict(list)
    for n in all_notes:
        if hasattr(n, 'pitch'):
            by_offset[n.offset].append(
                (n.pitch.midi, float(n.duration.quarterLength))
            )
        elif hasattr(n, 'pitches'):
            for p in n.pitches:
                by_offset[n.offset].append(
                    (p.midi, float(n.duration.quarterLength))
                )

    melody = []
    for offset in sorted(by_offset.keys()):
        notes_at = by_offset[offset]
        long_enough = [(p, d) for p, d in notes_at if d >= MIN_DURATION]
        if not long_enough:
            long_enough = notes_at
        melody.append(max(long_enough, key=lambda x: x[0])[0])

    return melody

for midi_path, title, composer in MIDI_FILES:
    print(f"\n{'='*60}")
    print(f"{composer} — {title}")
    
    try:
        melody = extract_melody(midi_path)
        intervals = [melody[i+1] - melody[i] for i in range(len(melody)-1)]
        folded = fold_octaves(intervals)
        
        print(f"Notes: {len(melody)}, Intervals: {len(folded)}")
        print(f"First 15 notes: {[note_number_to_name(p) for p in melody[:15]]}")
        print(f"First 20 folded: {folded[:20]}")
        
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n{'='*60}")
print("Done.")

import pretty_midi

def save_melody_midi(melody_pitches, output_path, tempo=120):
    """Save a list of MIDI pitches as a simple melody MIDI file."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # piano
    
    time = 0.0
    note_duration = 0.5  # each note half a second
    
    for pitch in melody_pitches:
        note = pretty_midi.Note(
            velocity=80,
            pitch=int(pitch),
            start=time,
            end=time + note_duration
        )
        instrument.notes.append(note)
        time += note_duration
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"Saved: {output_path}")

# extract eine kleine melody and save
score = converter.parse('data/midi/einekleine.mid')
part = score.parts[0]
all_notes = part.flatten().notes

by_offset = defaultdict(list)
for n in all_notes:
    if hasattr(n, 'pitch'):
        by_offset[n.offset].append((n.pitch.midi, float(n.duration.quarterLength)))
    elif hasattr(n, 'pitches'):
        for p in n.pitches:
            by_offset[n.offset].append((p.midi, float(n.duration.quarterLength)))

melody = []
for offset in sorted(by_offset.keys()):
    notes_at = by_offset[offset]
    long_enough = [(p, d) for p, d in notes_at if d >= 0.1]
    if not long_enough:
        long_enough = notes_at
    melody.append(max(long_enough, key=lambda x: x[0])[0])

save_melody_midi(melody[:50], "data/audio/eine_kleine_extracted.mid")