#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import PIL.Image,PIL.ImageTk
from tkinter import *  # 导入 Tkinter 库
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import yin
import write2midi
import pygame
from record_play import Recorder
from playsound import playsound

default_file = "scale.wav"
file_name = default_file

midi_file_name = "new_song.mid"


import wave
from pyaudio import PyAudio,paInt16


def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def my_record():
    pa=PyAudio()
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0
    while count<TIME*10:#控制录音时间
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count+=1
        print('.')
    save_wave_file('record.wav',my_buf)
    stream.close()
    print('Over!')


def play():
    chunk = 2014
    filename = entry_file_name.get()
    wf=wave.open(r"record.wav",'rb')
    p=PyAudio()
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=
    wf.getnchannels(),rate=wf.getframerate(),output=True)
    while True:
        data=wf.readframes(chunk)
        if data=="":break
        stream.write(data)
    stream.close()
    p.terminate()
    print('Over!')



def open_wav_file():
    file_name = askopenfilename()
    print(file_name)
    entry_file_name.delete(0, 'end')
    entry_file_name.insert(0, file_name)


r = Recorder()

def record():
    r.start()
    state_message.set("Recording!")
    # record_button.configure(text="stop")  # 设置button显示的内容

def stop_record():
    default_file_name = "record.wav"
    r.stop()
    r.save(default_file_name)
    entry_file_name.delete(0, END)
    entry_file_name.insert(0,default_file_name)
    state_message.set("Finished Recording!")


def play_wav():
    if "record" in entry_file_name.get():
        r.play()
    else:

        playsound( entry_file_name.get())

def start():
    w_len = int(entry_windows_length.get())
    w_step = int(entry_windows_step.get())
    min_f0 = int(entry_f0_min.get())
    max_f0 = int(entry_f0_max.get())
    t_thresold = float(entry_t_threshold.get())
    instrument = int(entry_instrument.get())
    file_name = entry_file_name.get()
    yin.compute(audioFileName=file_name, w_len=w_len, w_step=w_step, f0_min=min_f0, f0_max=max_f0, t_thresh=t_thresold)
    imLabel.config(image=None)
    imLabel.image = None
    img = PIL.Image.open("./visualization.jpg")
    img = img.resize((480, 360))
    img = PIL.ImageTk.PhotoImage(img)
    imLabel.config(image=img)
    imLabel.image = img

    state_message.set("The midi has been generated!")
    write2midi.midi(instrument, filename=midi_file_name)
    print("finished!")


def play_midi():
    state_message.set("playing music")
    file = midi_file_name
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1)
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(file)
    except:
        import traceback
        print(traceback.format_exc())
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)


root = Tk()  # 创建窗口对象的背景色
root.title("Solo")
root.geometry("800x500")

txt_fp = Label(root,text="File path:")
txt_windows_length = Label(root, text="windows_length(Samples)")
txt_windows_step = Label(root, text="window step")
txt_fre_min = Label(root, text="f0 min")
txt_fre_max = Label(root, text="f0 max")
txt_t_threshold = Label(root, text="time thresold for a note(s)")
txt_instrument = Label(root, text="Instrument(1-128)")


entry_file_name = Entry(root)
entry_windows_length = Entry(root, text="1024")
entry_windows_length.insert(0,"1024")
entry_windows_step = Entry(root, text="256")
entry_windows_step.insert(0,"256")
entry_f0_min = Entry(root, text="80")
entry_f0_min.insert(0,"80")
entry_f0_max = Entry(root, text="1100")
entry_f0_max.insert(0, "1100")
entry_t_threshold = Entry(root,text="0.1")
entry_t_threshold.insert(0,"0.1")
entry_instrument = Entry(root, text="1")
entry_instrument.insert(0,"1")

# entry_instrument = ttk.Combobox(root, width=18)
# entry_instrument['values'] = (1,2,3,4,5,6,7,8,9)     # 设置下拉列表的值
# entry_instrument.grid(column=1, row=1)      # 设置其在界面中出现的位置  column代表列   row 代表行
# entry_instrument.current(0)    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值



button_file = Button(root, text ="open wav file", command = open_wav_file, activeforeground='red', state="normal",	\
             activebackground = 'blue')

button_play = Button(root, text="play midi", command=play_midi)
start_button = Button(root, text="Start", command = start)
record_button = Button(root, text="record", command = record)
stop_record_button = Button(root, text="stop record", command = stop_record)
play_wav = Button(root,text="play wav",command=play_wav)

txt_windows_length.place(x=20,y=25)
entry_windows_length.place(x=20,y=45)
txt_windows_step.place(x=20,y=75)
entry_windows_step.place(x=20,y=95)
txt_fre_min.place(x=20,y=125)
entry_f0_min.place(x=20,y=145)
txt_fre_max.place(x=20,y=175)
entry_f0_max.place(x=20,y=195)
txt_t_threshold.place(x=20,y=225)
entry_t_threshold.place(x=20,y=245)
txt_instrument.place(x=20,y=275)
entry_instrument.place(x=20,y=295)

img = PIL.Image.open("./background.jpg")
img = img.resize((480, 360))
img = PIL.ImageTk.PhotoImage(img)
imLabel = Label(root, image=img)
imLabel.image= img
# imLabel.grid(row=0, column=3, rowspan=8, columnspan=5)
x_offset = 40
y_offset = 120
imLabel.place(x=300,y=20)

txt_fp.place(x=20,y=350)
entry_file_name.place(x=100,y=350)
button_file.place(x=200,y=350)

record_button.place(x=260+x_offset,y=275+y_offset)
stop_record_button.place(x=260+x_offset,y=310+y_offset)
play_wav.place(x=330+x_offset,y=275+120)
stop_record_button.place(x=260+x_offset,y=310+y_offset)
start_button.place(x=430+x_offset,y=275+y_offset)
button_play.place(x=490+x_offset,y=275+y_offset)

state_message = StringVar()
state_message.set("Please go!")
txt_state = Label(root, textvariable=state_message)
txt_state.place(x=430+x_offset,y=310+y_offset)
root.mainloop()  # 进入消息循环