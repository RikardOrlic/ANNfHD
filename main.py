from tkinter import *
from tkinter import ttk
import numpy as np
from ann import *



def paint(event):
    print(event.x, event.y)
    python_green = "#476042"
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    draw.create_oval(x1, y1, x2, y2, fill = python_green)
    global tmp
    tmp.append((event.x, event.y))

def clear(event):
    draw.delete('all')

def mmove(event):
    global tmp
    global tmpList
    global idx
    global y
    if tmp:
        tmpList.append(tmp.copy())
        tmp = []
        y_ = np.zeros(5)
        y_[idx] = 1
        y.append(y_.copy())#ne treba?
        item = tree.get_children()[-1]
        values_ = tree.item(item, 'values')
        tree.item(item, values=(values_[0], int(values_[1]) + 1))

def paint2(event):
    python_green = "#476042"
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    draw2.create_oval(x1, y1, x2, y2, fill = python_green)
    global gesta_
    gesta_.append((event.x, event.y))

def mmove2(event):
    global gesta_
    global ann
    if gesta_:
        outputTxt.delete(1.0, END)
        y_pred = ann.predict(np.reshape(np.array(reprez(gesta_)), (1, M*2)))[-1]
        outputTxt.insert(INSERT, "alpha={}, beta={}, gamma={}, delta={}, epsilon={}".format(*y_pred[0, :]))
        gesta_ = list()

def clear2(event):
    draw2.delete('all')
        

def load_data():
    global znakovi
    global koordinate
    global y
    koordinate = []
    y = []
    for idx_ in range(len(znakovi)):
        try:
            with open(znakovi[idx_] + '.txt', "r") as f:
                tmp = []
                for line in f.readlines():
                    line = line.rstrip()
                    x_, y_ = line.split(";")
                    y.append(np.array(y_.split(','), dtype='float64'))
                    trajektorija = []
                    for tocka in x_.split():
                        koord_x, koord_y = tocka.split(',')
                        trajektorija.append((koord_x, koord_y))
                    tmp.append(trajektorija)
                koordinate.append(tmp)        
        
        except IOError:
            print("IOError")

def save_data():
    global znakovi
    global koordinate
    global y
    count = 0
    for idx_ in range(len(znakovi)):
        try:
            with open(znakovi[idx_] + '.txt', "w") as f:
                for gesta in koordinate[idx_]:
                    tmp = ''
                    for tocka in gesta:
                        tmp += str(tocka[0]) + ',' + str(tocka[1]) + ' '
                    tmp = tmp[:-1] + ';'
                    for izlaz in y[count]:
                        tmp += str(izlaz) + ','
                    tmp = tmp[:-1] + '\n'
                    count += 1
                    f.write(tmp)
        except IOError:
            print("IOError")

def reprez(trajektorija):
    arr = np.array(trajektorija, dtype='float64')
    avg_x, avg_y = np.mean(arr, axis=0)
    arr -= (avg_x, avg_y)
    arr /= np.max(arr)
    d = 0
    d_od_nulte_tocke = [0]
    for row in range(1, np.shape(arr)[0]):
        d += np.linalg.norm(arr[row, :] - arr[row-1, :])
        d_od_nulte_tocke.append(d)
    d_od_nulte_tocke = np.array(d_od_nulte_tocke)
    uredeni_arr = [arr[0, :]]
    for k in range(1, M-1):
        i = np.argwhere(d_od_nulte_tocke > (k*d)/(M-1))[0]
        uredeni_arr.append(np.reshape((arr[i-1, :] + arr[i, :])/2, (2,)))
        
    uredeni_arr.append(arr[-1, :])
    return uredeni_arr
        


def next_sign():
    global idx
    global tmpList
    global koordinate
    global znakovi
    idx += 1
    tmpSave = []
    for trajektorija in tmpList:
        #pretvaranje trajektorije u 20 tocaka
        tmpSave.append(reprez(trajektorija))        

    koordinate.append(tmpSave)
    tmpList = []
    if idx < len(znakovi):
        values_= (znakovi[idx], 0)
        tree.insert('', END, values=values_)

    
def learn():
    global ann
    global koordinate
    global y
    tmp = []
    for i in range(len(koordinate)):
        for j in range(len(koordinate[i])):
            tmp.append(np.array(koordinate[i][j], dtype='float64').flatten())
    x = np.vstack(tmp)
    y_true = np.vstack(y)
    architecture = arch_value.get("1.0",END).rstrip()
    lr = float(eta_value.get("1.0",END).rstrip())
    cutoff_err = float(err_value.get("1.0",END).rstrip())
    max_iter = int(miter_value.get("1.0",END).rstrip())
    if option_variable.get() == 'batch':
        batch_size = np.shape(x)[0]
    elif option_variable.get() == 'mini-batch':
        batch_size = int(bsize_value.get("1.0",END).rstrip())
    else:
        batch_size = 1

    architecture = str(M*2) + ',' + architecture + ',' + str(5)
    ann = ANN(architecture, lr, cutoff_err, max_iter, batch_size, M)
    ann.fit(x, y_true)


if __name__ == '__main__':
    idx = 0
    znakovi = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    koordinate = []
    y = []
    tmpList = []
    tmp = []
    M = 10
    gesta_ = []
    

    mainWindow = Tk()
    mainWindow.geometry("1600x800")
    parent_tab = ttk.Notebook(mainWindow)

    crtanje_tab = ttk.Frame(parent_tab)
    ucenje_tab = ttk.Frame(parent_tab)
    testiranje_tab = ttk.Frame(parent_tab)

    #crtanje tab
    draw = Canvas(crtanje_tab)
    draw.pack(expand=1, fill='both', side=LEFT)
    draw.bind("<B1-Motion>", paint)
    draw.bind("<Motion>", mmove)
    draw.bind("<Button-3>", clear)
    
    slj_znak = Button(crtanje_tab, text="Sljedeci znak", command=next_sign)
    ucitaj = Button(crtanje_tab, text="Ucitaj podatke", command=load_data)
    spremi = Button(crtanje_tab, text="Spremi podatke", command=save_data)

    slj_znak.pack(side=LEFT, fill='y')
    ucitaj.pack(side=BOTTOM, fill='y')
    spremi.pack(side=BOTTOM, fill='y')
    
    columns = ('znak', 'broj_slika')
    tree = ttk.Treeview(crtanje_tab, columns=columns, show='headings')
    tree.heading('znak', text='Znak')
    tree.heading('broj_slika', text='Broj slika')
    tree.pack(side=LEFT, expand=1, fill='both')

    values_ = (znakovi[idx], 0)
    tree.insert('', END, values=values_)


    #ucenje_tab
    option_text = Text(ucenje_tab, width=30, height=1)
    option_text.insert(INSERT, "Nacin ucenja:")
    option_text.config(state=DISABLED)
    option_text.pack()

    OPTIONS = [
        "batch",
        "stohasticki",
        "mini-batch"
    ]
    option_variable = StringVar(ucenje_tab)
    option_variable.set(OPTIONS[0])
    option = OptionMenu(ucenje_tab, option_variable, *OPTIONS)
    option.pack()

    eta_text = Text(ucenje_tab, width=30, height=1)
    eta_text.insert(INSERT, "Eta:")
    eta_text.config(state=DISABLED)
    eta_text.pack()

    eta_value = Text(ucenje_tab, width=30, height=1)
    eta_value.insert(INSERT, "0.01")
    eta_value.pack()

    arch_text = Text(ucenje_tab, width=30, height=1)
    arch_text.insert(INSERT, "Arhitektura:")
    arch_text.config(state=DISABLED)
    arch_text.pack()

    arch_value = Text(ucenje_tab, width=30, height=1)
    arch_value.insert(INSERT, "5")
    arch_value.pack()

    err_text = Text(ucenje_tab, width=30, height=1)
    err_text.insert(INSERT, "Cuttof error:")
    err_text.config(state=DISABLED)
    err_text.pack()

    err_value = Text(ucenje_tab, width=30, height=1)
    err_value.insert(INSERT, "0.0001")
    err_value.pack()

    bsize_text = Text(ucenje_tab, width=30, height=1)
    bsize_text.insert(INSERT, "Batch size(ako mini-batch):")
    bsize_text.config(state=DISABLED)
    bsize_text.pack()

    bsize_value = Text(ucenje_tab, width=30, height=1)
    bsize_value.insert(INSERT, "10")
    bsize_value.pack()

    miter_text = Text(ucenje_tab, width=30, height=1)
    miter_text.insert(INSERT, "Max epoha:")
    miter_text.config(state=DISABLED)
    miter_text.pack()

    miter_value = Text(ucenje_tab, width=30, height=1)
    miter_value.insert(INSERT, "10000")
    miter_value.pack()

    #self.myText_Box.get("1.0",END)

    uci_BTN = Button(ucenje_tab, text="Nauci", command=learn)
    uci_BTN.pack()




    #testiranje_tab
    draw2 = Canvas(testiranje_tab)
    draw2.pack(expand=1, fill='both')
    draw2.bind("<B1-Motion>", paint2)
    draw2.bind("<Motion>", mmove2)
    draw2.bind("<Button-3>", clear2)

    outputTxt = Text(testiranje_tab, height=1)
    outputTxt.pack(fill='x', side=BOTTOM)




    parent_tab.add(crtanje_tab, text="Crtanje")
    parent_tab.add(ucenje_tab, text="Ucenje")
    parent_tab.add(testiranje_tab, text="Testiranje")

    parent_tab.pack(expand=1, fill='both')


    
    mainWindow.mainloop()

