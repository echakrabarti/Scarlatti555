from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from pretty_midi import note_number_to_name
import numpy as np

import pretty_midi

AUDIO_FILES = [
    "data/audio/fur_elise.mp3",
    "data/audio/nocturne.mp3",
    "data/audio/clair_de_lune.mp3",
    "data/audio/air_on_g_string.mp3",
    "data/audio/the_swan.mp3",
]

MIN_DURATION = 0.08
MIN_PITCH = 48    # C3 — nothing below this is melody
MAX_PITCH = 96    # C7 — nothing above this is melody

def save_midi(melody_notes, output_path):
    """Save extracted melody notes as a MIDI file you can play."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # piano
    
    for start, end, pitch, vel, _ in melody_notes:
        note = pretty_midi.Note(
            velocity=80,
            pitch=int(pitch),
            start=float(start),
            end=float(end)
        )
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"  Saved MIDI to {output_path}")


def fold_octaves(intervals, threshold=6):
    folded = []
    for interval in intervals:
        while interval > threshold:
            interval -= 12
        while interval < -threshold:
            interval += 12
        folded.append(interval)
    return folded

def skyline(notes):
    # sort by start time
    notes.sort(key=lambda n: n[0])
    
    result = []
    i = 0
    while i < len(notes):
        # group notes starting within 50ms of each other
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and abs(notes[j][0] - notes[i][0]) < 0.05:
            group.append(notes[j])
            j += 1
        # keep highest pitch in this time slice
        result.append(max(group, key=lambda n: n[2]))
        i = j
    
    return result

for audio_file in AUDIO_FILES:
    print(f"\n{'='*60}")
    print(f"File: {audio_file}")
    
    try:
        _, _, note_events = predict(audio_file, ICASSP_2022_MODEL_PATH)
        
        # step 1 — filter by duration and pitch range
        filtered = [
            n for n in note_events
            if (n[1] - n[0]) >= MIN_DURATION
            and MIN_PITCH <= int(n[2]) <= MAX_PITCH
        ]
        
        # step 2 — skyline: keep highest note at each timestep
        melody = skyline(filtered)
        save_midi(melody, f"data/audio/{audio_file.split('/')[-1].replace('.mp3', '_melody.mid')}")
        print(f"Notes: {len(note_events)} raw → {len(filtered)} range filtered → {len(melody)} after skyline")
        
        print("First 15 melody notes:")
        for start, end, pitch, vel, _ in melody[:15]:
            name = note_number_to_name(int(pitch))
            print(f"  {name:<4} (midi {int(pitch)})  dur={end-start:.2f}s")
        
        pitches = [int(n[2]) for n in melody]
        raw_intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        folded = fold_octaves(raw_intervals)
        
        print(f"Total intervals: {len(folded)}")
        print(f"First 20 (folded): {folded[:20]}")
        
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n{'='*60}")
print("Done.")

