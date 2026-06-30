from music21 import converter

VOCAL_MIDI_FILES = [
    "data/PhidyleMIDIs/ave_maria_(c)yogore.mid",
    "data/PhidyleMIDIs/hirt_auf_dem_felsen_129_1_(c)bakels.mid",  # confirm full name
    "data/PhidyleMIDIs/ruh_(c)yogore.mid",
    "data/PhidyleMIDIs/winterreise_1_(c)yogore.mid",
    "data/PhidyleMIDIs/winterreise_7_(c)yogore.mid",
]

for path in VOCAL_MIDI_FILES:
    print(f"\n{'='*60}")
    print(f"File: {path}")
    
    try:
        score = converter.parse(path)
        print(f"Number of parts: {len(score.parts)}")
        
        for i, part in enumerate(score.parts):
            inst = part.getInstrument()
            notes = part.flatten().notes
            print(f"  Part {i}: {inst} — {len(notes)} notes")
            
            pitches = []
            for n in notes[:10]:
                if hasattr(n, 'pitch'):
                    pitches.append(str(n.pitch))
                elif hasattr(n, 'pitches'):
                    pitches.append(str(n.pitches[0]))
            print(f"    First notes: {pitches}")
    
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n{'='*60}")
print("Done.")