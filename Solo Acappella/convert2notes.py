

def generate():


    # C0
    base = [15.892725498, 16.351597831, 16.837756133]
    # 12th root of 2
    multiplier = 1.05946309436

    notes = {12: base}
    for i in range(13, 127):
        mid = multiplier * notes[i - 1][1]
        low = (mid + notes[i - 1][1]) / 2.0
        high = (mid + (multiplier * mid)) / 2.0
        notes.update({i : [low, mid, high]})

    return notes

def readFreqs(file):
    f = open(file, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    freqs = []
    for line in lines:
        res = line.split("\n")[0]
        result = res.split('\t', 2)
        freqs.append(result)
    return freqs


def freq2notes(notes,freqs):

    newNotes = []

    for freq in freqs:
        note = []
        duration = float(freq[2]) - float(freq[0])

        for key in notes:
            if notes[key][0] <= float(freq[1]) <= notes[key][2]:
                if key in notes.keys():
                    note.append(key)
                    note.append(int(duration * 1000))

        newNotes.append(note)

    return newNotes

