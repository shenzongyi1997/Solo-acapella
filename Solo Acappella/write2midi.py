from mido import Message, MidiFile, MidiTrack
import convert2notes as cn

def midi(num, filename):

    freqs = cn.readFreqs("notes.txt")
    notes = cn.generate()
    newNotes = cn.freq2notes(notes,freqs)

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=num, time=0))

    for newNote in newNotes:
        track.append(Message('note_on', note=newNote[0], velocity=64, time=newNote[1]))
        track.append(Message('note_off', note=newNote[0], velocity=64, time=newNote[1]))

    mid.save(filename)



