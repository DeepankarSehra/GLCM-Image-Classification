from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageTk
from algorithm import Algorithm
# Create an instance of tkinter frame
window = Tk()
# Set the geometry of tkinter frame
window.geometry("400x350")

def openFile():     # Function for choosing input image, GUI lets you choose directory too 
    global my_image
    filepath = filedialog.askopenfilename(initialdir="D:\\EE769\\project",
                                          title="Select Image File",
                                          filetypes=(("jpeg files", "*.jpg"), ("All Files", "*.*")))
    # my_image = filepath
    my_image = filepath
    # my_image.show() # Display the selected image in a new window

def processImage():
    global my_image
    global k            # Number of clusters for K-Means 
    global outpath
    try:
        n = int(text_entry.get())
        k = n
        # Process the image here (using the selected file path and the entered integer)

        # Save the output image, lets you choose directory and name
        savePath = filedialog.asksaveasfilename(initialdir="D:\\EE769\\project",
                                                title="Save Output Image",
                                                filetypes=(("jpeg files", "*.jpg"), ("All Files", "*.*")))
        if savePath:
            outpath = savePath
            messagebox.showinfo("Success", "Output will be saved here")
        window.destroy()
    except ValueError:
        # If the entered value is not a valid integer, show an error message
        error_label.config(text="Error: Please enter a valid integer")

my_label = Label(window, text="Select Image for Textural classification", font=('Times', 15))
my_label.pack(pady=20)

button = Button(window, text="Click to select Image", command=openFile)
button.pack(pady=20)

text_label = Label(window, text="Enter an integer:")    # For number of clusters
text_label.pack()

text_entry = Entry(window)
text_entry.pack()

error_label = Label(window, fg="red")
error_label.pack()

process_button = Button(window, text="Process Image", command=processImage)
process_button.pack(pady=20)


window.mainloop()

Algorithm(my_image, k, outpath)
print("yum ", my_image, k,outpath)
