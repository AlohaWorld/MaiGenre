#midi编码
#半参考
import os
import pathlib
import glob
from music21 import converter, instrument, note, chord, stream
midi_notes = []
DATA_DIR = "./data/"

for i, file in enumerate(glob.glob(os.path.join(DATA_DIR, "*.mid"))):
    midi_p = pathlib.Path(file)
    midi_file_name = midi_p.stem

    midi = converter.parse(file)
    print('\r', 'Parsing file ', i, " ", file, end='')

    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            midi_notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            midi_notes.append('.'.join(str(n) for n in element.normalOrder))

