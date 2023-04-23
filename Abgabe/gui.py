import tkinter as tk
import numpy as np
import cv2
from ctypes import *

c_double_p = POINTER(c_double)
c_double_pp = POINTER(c_double_p)
c_float_p = POINTER(c_float)
c_float_pp = POINTER(c_float_p)
c_float_ppp = POINTER(c_float_pp)
c_int_p = POINTER(c_int)


class setup(Structure):
    _fields_ = [("batch_size", c_int),
                ("n_train", c_int),
                ("n_test", c_int),
                ("epochs", c_int),
                ("learn_rate", c_float),
                ("n_layer", c_int),
                ("n_outputs", c_int),
                ("n_inputs", c_int),
                ("n_hidden_nodes", c_int)]


class network(Structure):
    _fields_ = [("mnist_data", c_double_pp),
                ("train_label_data", c_float_pp),
                ("test_label_data", c_float_pp),
                ("biases", c_float_pp),
                ("weights", c_float_ppp),
                ("learn_factors", c_float_pp),
                ("nodes", c_float_ppp),
                ("test_nodes", c_float_ppp),
                ("n_nodes", c_int_p)]


class gui(tk.Tk):
    #Predefined train values
    epochs = 10
    learn_rate = 0.01
    batch_size = 10
    n_train = 60000
    n_test = 10000
    n_layer = 3
    n_inputs = 784
    n_hidden = 16
    n_outputs = 10

    def __init__(self):
        tk.Tk.__init__(self)
        #Create Gui
        self.net = None
        self.stp = None
        self.stp_p = None
        self.net_p = None
        self.geometry("880x350")
        self.title("Neural Net")
        self.config(bg='#c6e2ff')
        self.resizable(False, False)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=252, height=252, bd=0, bg="#93c8ff", cursor="cross", confine=True)
        self.canvas.grid(column=3, row=1, padx=10, pady=10, rowspan=5, columnspan=2)
        # Create Canvas for the results
        self.canvasTwo = tk.Canvas(self, width=252, height=252, bd=0, bg="#93c8ff", confine=True)
        self.canvasTwo.grid(column=5, row=1, padx=10, pady=10, rowspan=5, columnspan=2)
        self.nodeZero = self.canvasTwo.create_oval(90, 2, 110, 22, width=2, fill="red")
        self.canvasTwo.create_text(100, 14, text="0", fill="black", font='Helvetica 15 bold')
        self.txtZero = self.canvasTwo.create_text(160, 14, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 14, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 12, 90, 12, width=2)
        self.canvasTwo.create_line(0, 37, 90, 12, width=2)

        self.nodeOne = self.canvasTwo.create_oval(90, 27, 110, 47, width=2, fill="red")
        self.canvasTwo.create_text(100, 39, text="1", fill="black", font='Helvetica 15 bold')
        self.txtOne = self.canvasTwo.create_text(160, 39, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 39, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 12, 90, 37, width=2)
        self.canvasTwo.create_line(0, 37, 90, 37, width=2)
        self.canvasTwo.create_line(0, 62, 90, 37, width=2)

        self.nodeTwo = self.canvasTwo.create_oval(90, 52, 110, 72, width=2, fill="red")
        self.canvasTwo.create_text(100, 64, text="2", fill="black", font='Helvetica 15 bold')
        self.txtTwo = self.canvasTwo.create_text(160, 64, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 64, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 37, 90, 62, width=2)
        self.canvasTwo.create_line(0, 62, 90, 62, width=2)
        self.canvasTwo.create_line(0, 87, 90, 62, width=2)

        self.nodeThree = self.canvasTwo.create_oval(90, 77, 110, 97, width=2, fill="red")
        self.canvasTwo.create_text(100, 89, text="3", fill="black", font='Helvetica 15 bold')
        self.txtThree = self.canvasTwo.create_text(160, 89, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 89, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 62, 90, 87, width=2)
        self.canvasTwo.create_line(0, 87, 90, 87, width=2)
        self.canvasTwo.create_line(0, 112, 90, 87, width=2)

        self.nodeFour = self.canvasTwo.create_oval(90, 102, 110, 122, width=2, fill="red")
        self.canvasTwo.create_text(100, 114, text="4", fill="black", font='Helvetica 15 bold')
        self.txtFour = self.canvasTwo.create_text(160, 114, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 114, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 87, 90, 112, width=2)
        self.canvasTwo.create_line(0, 112, 90, 112, width=2)
        self.canvasTwo.create_line(0, 137, 90, 112, width=2)

        self.nodeFive = self.canvasTwo.create_oval(90, 127, 110, 147, width=2, fill="red")
        self.canvasTwo.create_text(100, 139, text="5", fill="black", font='Helvetica 15 bold')
        self.txtFive = self.canvasTwo.create_text(160, 139, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 139, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 112, 90, 137, width=2)
        self.canvasTwo.create_line(0, 137, 90, 137, width=2)
        self.canvasTwo.create_line(0, 162, 90, 137, width=2)

        self.nodeSix = self.canvasTwo.create_oval(90, 152, 110, 172, width=2, fill="red")
        self.canvasTwo.create_text(100, 164, text="6", fill="black", font='Helvetica 15 bold')
        self.txtSix = self.canvasTwo.create_text(160, 164, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 164, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 137, 90, 162, width=2)
        self.canvasTwo.create_line(0, 162, 90, 162, width=2)
        self.canvasTwo.create_line(0, 187, 90, 162, width=2)

        self.nodeSeven = self.canvasTwo.create_oval(90, 177, 110, 197, width=2, fill="red")
        self.canvasTwo.create_text(100, 189, text="7", fill="black", font='Helvetica 15 bold')
        self.txtSeven = self.canvasTwo.create_text(160, 189, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 189, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 162, 90, 187, width=2)
        self.canvasTwo.create_line(0, 187, 90, 187, width=2)
        self.canvasTwo.create_line(0, 212, 90, 187, width=2)

        self.nodeEight = self.canvasTwo.create_oval(90, 202, 110, 222, width=2, fill="red")
        self.canvasTwo.create_text(100, 214, text="8", fill="black", font='Helvetica 15 bold')
        self.txtEight = self.canvasTwo.create_text(160, 214, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 214, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 187, 90, 212, width=2)
        self.canvasTwo.create_line(0, 212, 90, 212, width=2)
        self.canvasTwo.create_line(0, 237, 90, 212, width=2)

        self.nodeNine = self.canvasTwo.create_oval(90, 227, 110, 247, width=2, fill="red")
        self.canvasTwo.create_text(100, 239, text="9", fill="black", font='Helvetica 15 bold')
        self.txtNine = self.canvasTwo.create_text(160, 239, text="0", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_text(200, 239, text="%", fill="black", font='Helvetica 15 bold')
        self.canvasTwo.create_line(0, 212, 90, 237, width=2)
        self.canvasTwo.create_line(0, 237, 90, 237, width=2)

        # Buttons to test the neural net and clear the canvas
        self.labelAnswer = tk.Label(self, text=" ",
                                    bg='#c6e2ff')
        self.labelAnswer.grid(column=5, row=6, columnspan=2)
        self.testNet = tk.Button(self, text="Test Neural Net", command=self.calltestNet, bg='#7abbff',state="disabled")
        self.testNet.grid(column=3, row=6, pady=10)
        self.button_clear = tk.Button(self, text="Clear Canvas", command=self.clear_all, bg='#7abbff')
        self.button_clear.grid(column=4, row=6, pady=10)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)

        # Entry fields for the parameters
        self.labelOne = tk.Label(self, text="Epochs:", bg='#c6e2ff')
        self.labelOne.grid(column=0, row=1, padx=10, pady=10)
        self.labelTwo = tk.Label(self, text="Learn-Rate:", bg='#c6e2ff')
        self.labelTwo.grid(column=0, row=2, padx=10, pady=10)
        self.labelThree = tk.Label(self, text="Batch-size:", bg='#c6e2ff')
        self.labelThree.grid(column=0, row=3, padx=10, pady=10)
        self.labelFour = tk.Label(self, text="No. Layers:", bg='#c6e2ff')
        self.labelFour.grid(column=0, row=4, padx=10, pady=10)
        self.labelFive = tk.Label(self, text="No. Hidden:", bg='#c6e2ff')
        self.labelFive.grid(column=0, row=5, padx=10, pady=10)
        self.entryOne = tk.Entry(self, width=20)
        self.entryOne.insert(-1, str(self.epochs))
        self.entryOne.grid(column=1, row=1, padx=10, pady=10)
        self.entryTwo = tk.Entry(self, width=20)
        self.entryTwo.insert(-1, str(self.learn_rate))
        self.entryTwo.grid(column=1, row=2, padx=10, pady=10)
        self.entryThree = tk.Entry(self, width=20)
        self.entryThree.insert(-1, str(self.batch_size))
        self.entryThree.grid(column=1, row=3, padx=10, pady=10)
        self.entryFour = tk.Entry(self, width=20)
        self.entryFour.insert(-1, str(self.n_layer))
        self.entryFour.grid(column=1, row=4, padx=10, pady=10)
        self.entryFive = tk.Entry(self, width=20)
        self.entryFive.insert(-1, str(self.n_hidden))
        self.entryFive.grid(column=1, row=5, padx=10, pady=10)
        self.buttonTrain = tk.Button(self, text="Train Neural Net", bg='#7abbff', command=self.TrainNeuralNet)
        self.buttonTrain.grid(column=0, row=6, rowspan=2, pady=10, padx=4)
        self.labelTrain = tk.Label(self, text=" ", bg='#c6e2ff', fg='green')
        self.labelTrain.grid(column=1, row=6, pady=10, padx=5)

    def clear_all(self):
        #Delete the drawing on the canvas and clear the list with the coordinates
        self.canvas.delete("all")
        self.points_recorded.clear()

    def TrainNeuralNet(self):
        #Train the neural Net
        epochs = int(self.entryOne.get()) # Get values from the entries
        learn_rate = float(self.entryTwo.get())
        batch_size = int(self.entryThree.get())
        n_layer = int(self.entryFour.get())
        n_hidden = int(self.entryFive.get())

        self.stp_p = POINTER(setup)
        # arguments for stp init (the settings for the net)
        netPy.stpInit.argtypes = [c_int, c_float, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        # return value is stp*
        netPy.stpInit.restype = self.stp_p

        self.stp = netPy.stpInit(epochs, learn_rate, batch_size, n_layer, n_hidden, self.n_test, self.n_train,
                                 self.n_inputs, self.n_outputs)
        self.net_p = POINTER(network)
        # initNet args
        netPy.netInit.argtypes = [self.stp_p]
        # returns
        netPy.netInit.restype = self.net_p
        # init struct Net
        self.net = netPy.netInit(self.stp)
        # learning
        # args and return
        netPy.pyTrainMnist.argtypes = [self.stp_p, self.net_p]
        netPy.pyTrainMnist.restype = self.net_p
        self.net = netPy.pyTrainMnist(self.stp, self.net)
        #show success message
        self.labelTrain['text'] = "Training completed!"
        self.testNet['state'] = "normal" #enable the test net button


    def calltestNet(self):
        #call function
        self.TestNeuralNet(self.net)

    def TestNeuralNet(self, net):
        #Test the neural net with the drawing of the canvas
        x = 252
        y = 252
        zero = np.zeros(shape=(y, x), dtype=np.uint8)  # create 252x252 array with only zeros in it
        ys = self.points_recorded[0::2]  # get all the y coordinates from the list
        xs = self.points_recorded[1::2]  # get all the x coordinates from the list
        for i, j in zip(xs, ys):  # iterate through both arrays, to get the coordinates where the user had drawn
            zero[i, j] = 255
            zero[i + 1, j] = 255  # also marking the surrounding pixels as painted to get a thicker line
            zero[i - 1, j] = 255

            zero[i, j + 1] = 255
            zero[i + 1, j + 1] = 255
            zero[i - 1, j + 1] = 255

            zero[i, j - 1] = 255
            zero[i + 1, j - 1] = 255
            zero[i - 1, j - 1] = 255

        testarray = cv2.resize(zero, dsize=(28, 28),
                               interpolation=cv2.INTER_AREA)  # resizing the 512x512 array to a 28x28 array also using interpolation to get the grey values for the new array
        newlist = []

        for x in testarray:  # multiply every value by 2 to make the line thicker and then append the values to a 1-dimensional List
            for y in x:
                y *= 2
                if y > 255:
                    newlist.append(255)
                else:
                    newlist.append(y)

        newnumpy = np.array(newlist) #convert list to numpy array
        data = newnumpy / 255
        data = data.astype(np.float32)
        data_p = data.ctypes.data_as(c_float_p)
        net.contents.test_nodes[0][0] = data_p
        netPy.pyFeedForward.argtypes = [self.stp_p, self.net_p, c_int, c_int,
                                        c_int]  # last int 0 for train | 1 for test
        netPy.pyFeedForward.restype = self.net_p
        # call feedForwar for test
        net = netPy.pyFeedForward(self.stp, net, 0, self.stp.contents.n_test, 1)
        results = []
        for i in range(0, 10):
            results.append((net.contents.test_nodes[0][self.stp.contents.n_layer - 1][i]) * 100) #Test results saved in array
        highest = max(results) #get highest percentage
        highestindex = 0
        #make every oval if not yet red again
        self.canvasTwo.itemconfig(self.nodeZero, fill="red")
        self.canvasTwo.itemconfig(self.nodeOne, fill="red")
        self.canvasTwo.itemconfig(self.nodeTwo, fill="red")
        self.canvasTwo.itemconfig(self.nodeThree, fill="red")
        self.canvasTwo.itemconfig(self.nodeFour, fill="red")
        self.canvasTwo.itemconfig(self.nodeFive, fill="red")
        self.canvasTwo.itemconfig(self.nodeSix, fill="red")
        self.canvasTwo.itemconfig(self.nodeSeven, fill="red")
        self.canvasTwo.itemconfig(self.nodeEight, fill="red")
        self.canvasTwo.itemconfig(self.nodeNine, fill="red")

        #change oval color and percentage text according to the results list
        for i in range(0, 10):
            if i == 0:
                self.canvasTwo.itemconfig(self.txtZero, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeZero, fill="green")
                    highestindex = i
            if i == 1:
                self.canvasTwo.itemconfig(self.txtOne, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeOne, fill="green")
                    highestindex = i
            if i == 2:
                self.canvasTwo.itemconfig(self.txtTwo, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeTwo, fill="green")
                    highestindex = i
            if i == 3:
                self.canvasTwo.itemconfig(self.txtThree, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeThree, fill="green")
                    highestindex = i
            if i == 4:
                self.canvasTwo.itemconfig(self.txtFour, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeFour, fill="green")
                    highestindex = i
            if i == 5:
                self.canvasTwo.itemconfig(self.txtFive, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeFive, fill="green")
                    highestindex = i
            if i == 6:
                self.canvasTwo.itemconfig(self.txtSix, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeSix, fill="green")
                    highestindex = i
            if i == 7:
                self.canvasTwo.itemconfig(self.txtSeven, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeSeven, fill="green")
                    highestindex = i
            if i == 8:
                self.canvasTwo.itemconfig(self.txtEight, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeEight, fill="green")
                    highestindex = i
            if i == 9:
                self.canvasTwo.itemconfig(self.txtNine, text="{:.2f}".format(results[i]))
                if results[i] == highest:
                    self.canvasTwo.itemconfig(self.nodeNine, fill="green")
                    highestindex = i
        #Add an answer sentence
        answer = "The neural net thinks with a certainty\n of "
        answer += "{:.2f}".format(highest)
        answer += "% that you have drawn a "
        answer += str(highestindex)
        self.labelAnswer['text'] = answer

    def tell_me_where_you_are(self, event):
        #get coordinates of cursor
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        #draw on the canvas where the cursor is
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        if 0 < self.x < 252 and 0 < self.y < 252:  # add only the values within the defined border to the list
            self.canvas.create_line(self.previous_x, self.previous_y,
                                    self.x, self.y, fill="black")
            self.points_recorded.append(self.previous_x)
            self.points_recorded.append(self.previous_y)
            self.points_recorded.append(self.x)
            self.points_recorded.append(self.y)
            self.previous_x = self.x
            self.previous_y = self.y
        else:
            self.previous_x = self.x
            self.previous_y = self.y


if __name__ == "__main__":
    netPy = CDLL('./src/netPy.so')
    app = gui()
    app.mainloop()
