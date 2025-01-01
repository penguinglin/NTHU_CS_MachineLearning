# %%
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
        
# IMAGE_SIZE = 224 #EfficientNetB0, DenseNet121
IMAGE_SIZE = 448 # EfficientNetV2M

# %%
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageTk

# Scikit-learn includes many helpful utilities
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

# %%
datasets=['./offline-train.pkl','./train_caption.txt']
valid_datasets=['./offline-test.pkl', './test_caption.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=500000
valid_batch_Imagesize=500000
# batch_size for training and testing
batch_size=32
# the max (label length/Image size) in training and testing
# you can change 'maxlen','maxImagesize' by the size of your GPU
maxlen=48
maxImagesize= 100000
# hidden_size in RNN
hidden_size = 256
# teacher_forcing_ratio 
teacher_forcing_ratio = 1

AUG = False
AUG_NUM = 5

# %% [markdown]
# Data Augmentation

# %%
# import albumentations as A
from tqdm import tqdm
import cv2

# %% [markdown]
# Construct dictionaries

# %%
fp=open(dictionaries[0])

symbol_index={}

for line in fp.readlines():
    words = line.strip().split()
    symbol_index[words[0]]=int(words[1])  # word->index

fp.close()

print('total words/phones',len(symbol_index))

index_symbol={}
for k,v in symbol_index.items():
    index_symbol[v]=k  # index->word
    
print('total words/phones',len(index_symbol))


# %%
label_fp = open(datasets[1], 'r')
labels = label_fp.readlines()
label_fp.close()
label_num = len(labels)
print('total labels',label_num)

caption_list = []

for i in range(label_num):

    label = labels[i].strip().split()
    file_id = label[0]
    label = label[1:]
    caption_list.append(label)
        
# caption_list = np.array(caption_list)
print('caption_list[0]',caption_list[0])

# %%
img_name_list = []
# caption_list = []
i = 0
with open('./train_caption.txt') as fh:
    for line in fh:
        image_name = line.strip().split()[0]
        caption = caption_list[i]
        img_name_list.append(f'./off_image_train/{image_name}_0.jpg')
        caption_list[i] = ('<start> ' + ' '.join(caption) + ' <end>')
        i += 1
        
# Only first 120,000 images have labels

# test_img_name = set(glob(f'./words_captcha/*.png')) - set(img_name_list)
# img_name_list += sorted(test_img_name)

print(img_name_list[0])
print(caption_list[0]) 

# %%
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='', filters=' ')
# tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='')
tokenizer.fit_on_texts(caption_list)
caption_seq = tokenizer.texts_to_sequences(caption_list)

# iterate over the whole dataset if its caption over 48 words, remove it

tmp_image_name_list = []
tmp_caption_seq = []

for i in range(len(caption_seq)):
    if len(caption_seq[i]) <= 48:
        tmp_image_name_list.append(img_name_list[i])
        tmp_caption_seq.append(caption_seq[i])

img_name_list = tmp_image_name_list
caption_seq = tmp_caption_seq

caption_seq = tf.keras.preprocessing.sequence.pad_sequences(caption_seq, padding='post')
max_length = len(caption_seq[0])

print(caption_list[0])
print(caption_seq[0])
print (max_length)

# %%
img_name_train = img_name_list[:6500]
img_name_valid = img_name_list[6500:]

caption_seq_train = caption_seq[:6500]
caption_seq_valid = caption_seq[6500:]

# img_name_test = img_name_list[120000:]
# caption_seq_test = caption_seq[120000:]

# print the first caption and image name
print('caption_seq_train[0]',caption_seq_train[0])
print('img_name_train[0]',img_name_train[0])


# %%
import os
for i in range(len(img_name_train)):
    if not os.path.exists(img_name_train[i]):
        print('not exist',img_name_train[i])
        break

# %%
BATCH_SIZE = 16 # 100 is a suck number
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE

# %%
def load_image(image_path, caption):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    # Let's try to use padding instead of resize
    img = tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return image, caption

# %%
dataset_train = tf.data.Dataset.from_tensor_slices((img_name_train, caption_seq_train))\
                               .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                               .shuffle(BUFFER_SIZE)\
                               .batch(BATCH_SIZE, drop_remainder=True)\
                               .prefetch(tf.data.experimental.AUTOTUNE)

# %%
image_model = tf.keras.applications.EfficientNetV2M(include_top=False,
                                                weights='imagenet')
# image_model = tf.keras.applications.DenseNet121(include_top=False,
#                                                 weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

feature_extractor = tf.keras.Model(new_input, hidden_layer)

# %%
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# %%
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# %%
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(0.1)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden[0])

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the first GRU
        output, state = self.gru1(x, initial_state=hidden)

        # passing the output of the first GRU to the second GRU
        output, state = self.gru2(output, initial_state=state)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # applying dropout
        x = self.dropout(x)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, [state], attention_weights

    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units))]

# %%
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# %%
checkpoint_path = r".\checkpoints\train"
ckpt = tf.train.Checkpoint(feature_extractor=feature_extractor,
                           encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

# %%
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

ckpt_manager.restore_or_initialize()


# %%
from tqdm import tqdm

# %%
def predict(img_tensor):
    batch_size = img_tensor.shape[0]
    
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

    hidden = decoder.reset_state(batch_size=batch_size)
    
    features = feature_extractor(img_tensor)
    features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
    features = encoder(features)


    result = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)
    
    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        
        predicted_id = tf.argmax(predictions, axis=1).numpy()
        dec_input = tf.expand_dims(predicted_id, 1)
        result = tf.concat([result, predicted_id.reshape((batch_size, 1))], axis=1)

    return result

# %%
def build_output(args):
    results = []
    for i in args:
        result = ""
        for s in i[1:]:
            if s == tokenizer.word_index["<end>"]:
                break
            else :
                result += (tokenizer.index_word[s] + " ")
        results.append(result)
    return results



# %%
def map_test(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    # img = tf.keras.applications.inception_v3.preprocess_input(img)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return image, image_path

# %%
# img_name_test = ["./2+4.jpg"]

def evaluate_single(image_path):
    image, image_path = map_test(image_path)
    image = tf.expand_dims(image, 0)
    result = build_output(predict(image).numpy())
    print("result of transform is: ",result[0])
    return result[0]

# %%
from IPython.display import Latex
            
print("Openning the app")

import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import colorchooser, filedialog
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog
from PIL import ImageGrab
import mss


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("ML Final Project")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((1, 2), weight=1)
        self.grid_rowconfigure(0, weight=3)


        """
        # sidebar frame
            > Label 
            > Button 
                > Save - TODO(Function: Save_event)
                > Clean
                > Mode
                > Scaling
        """
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        # label - change name (optional)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Hello", font=customtkinter.CTkFont(family="Times New Roman",size=20, weight="bold",slant="italic",underline=False))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Save button
        self.save_button = customtkinter.CTkButton(master=self.sidebar_frame, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), hover_color="#b4b4b4", command=self.Save_event, text="SAVE")
        self.save_button.grid(row=1, column=0, padx=20, pady=10)
        
        # Clean button
        self.clean_button = customtkinter.CTkButton(master=self.sidebar_frame, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), hover_color="#b4b4b4", command=self.Clean_event, text="CLEAN")
        self.clean_button.grid(row=2, column=0, padx=20, pady=10)
        
        # mode
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        
        # scaling
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))


        """
        # Output box
            > Box (Modify output can call function: update_output())
            > Calculate - TODO(Function: Calculate_event)
        """
        # create main entry and button
        # self.output_label = customtkinter.CTkLabel(self, text="Output here", anchor="w")
        # self.output_label.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        # # How to modify: self.output_label.configure(text="新的輸出內容")
        # other choice - output box
        self.output_box = customtkinter.CTkTextbox(self, width=340, height=100, font=("Arial", 12, "bold","italic"))
        self.output_box.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.output_box.insert("0.0", "System Output Here...")
        self.output_box.configure(state="disabled")

        # Calculate
        self.Calculate_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),hover_color="#b4b4b4",text="Calculate",command=self.Calculate_event)
        self.Calculate_button.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        
        
        
        
        """
        # Canva
            nothing special
        """
        # Canva 
        self.canvas = customtkinter.CTkCanvas(self, width=400, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")  
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<Button-1>", self.set_start)
        self.canvas.bind("<ButtonRelease-1>", self.end_line)
        # Initialize line drawing parameters
        self.line_id = None
        self.line_points = []
        self.line_options = {"fill": "#FFFFFF", "width": 2}  # default line options
        # Drawing state
        self.pen_size = 7
        self.eraser_size = 10
        self.tool_mode = "pen"  # default to pen
        self.pen_color = "#FFFFFF"
        self.eraser_color = "#000000"
        
        
        
        """
        # Tabview
            > Draw Tool
                > Pen
                > Eraser
            > Tool Control
                > Pen slider
                > Eraser slider
        """
        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Draw Tool")
        self.tabview.tab("Draw Tool").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        # Create tool choice frame inside the "Draw Tool" tab
        self.tool_choice = customtkinter.CTkFrame(self.tabview.tab("Draw Tool"))
        self.tool_choice.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.tool_choice.grid_columnconfigure(0, weight=1)
        self.tool_var = tkinter.IntVar(value=0)
        # Pen button
        self.pen_button = customtkinter.CTkRadioButton(master=self.tool_choice, variable=self.tool_var, value=0, text="Pen", command=self.Draw_event)
        self.pen_button.grid(row=1, column=0, pady=10, padx=20, sticky="n")
        # Eraser button
        self.eraser_button = customtkinter.CTkRadioButton(master=self.tool_choice, variable=self.tool_var, value=1, text="Eraser", command=self.Erase_event)
        self.eraser_button.grid(row=2, column=0, pady=10, padx=20, sticky="n")
        # second control
        self.tabview.add("Tool Control")
        # Create a frame to hold sliders for Pen and Eraser size
        self.slider_frame = customtkinter.CTkFrame(self.tabview.tab("Tool Control"))
        self.slider_frame.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="nsew")
        self.slider_frame.grid_columnconfigure((0, 1), weight=1) 
        self.pen_label = customtkinter.CTkLabel(self.slider_frame, text="Pen Size:")
        self.pen_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="n")
        self.pen_slider = customtkinter.CTkSlider(
            self.slider_frame, orientation="vertical", from_=1, to=50, number_of_steps=49, command=self.update_pen_size
        )
        self.pen_slider.set(self.pen_size)
        self.pen_slider.grid(row=1, column=0, padx=10, pady=10, sticky="n")
        self.eraser_label = customtkinter.CTkLabel(self.slider_frame, text="Eraser Size:")
        self.eraser_label.grid(row=0, column=1, padx=10, pady=(10, 5), sticky="n")
        self.eraser_slider = customtkinter.CTkSlider(
            self.slider_frame, orientation="vertical", from_=1, to=50, number_of_steps=49, command=self.update_eraser_size
        )
        self.eraser_slider.set(self.eraser_size)
        self.eraser_slider.grid(row=1, column=1, padx=10, pady=10, sticky="n")        
        

        """
        Some default values
        """
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        


    """
    # event function
    """
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
        
    def draw_line(self, event):
        """ Draws a line or shapes depending on tool selection """
        x, y = event.x, event.y
        
        size = self.pen_size if self.tool_mode == "pen" else self.eraser_size
        color = self.pen_color if self.tool_mode == "pen" else "black"  
        
        self.line_points.extend((x, y))
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.line_points, width=size, fill=color) 

    def set_start(self, event):
        """ Sets the starting point for drawing """
        x, y = event.x, event.y
        self.line_points.clear()  # Reset line points before drawing new line
        self.line_points.extend((x, y))

    def end_line(self, event=None):
        """ Ends the line drawing session """
        self.line_id = None
        self.line_points.clear()
        
        
    def update_output(self, new_text):
            self.output_box.configure(state="normal")
            self.output_box.delete("0.0", "end") 
            self.output_box.insert("0.0", new_text)  
            self.output_box.configure(state="disabled")  
        
        
    def Save_event(self):
        # 用 postscript 生成 .eps 文件
        # ps_file = "canvas_output.ps"
        # self.canvas.postscript(file=ps_file, colormode='color')
        
        # img = Image.open("canvas_output.ps")
        # img.save("OutputFile.jpg", "JPEG")
        # print(f"Image saved")  
        # 取得畫布的範圍
        x1 = self.canvas.winfo_rootx()
        y1 = self.canvas.winfo_rooty()
        x2 = x1 + self.canvas.winfo_width()
        y2 = y1 + self.canvas.winfo_height()
        
        # 截圖並保存
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        screenshot.save("OutputFile.jpg", "JPEG")
        print(f"Image saved successfully as OutputFile.jpg")
    
    def render_latex(self,latex_str):
        bg = "White"
        ft = "black"
        if customtkinter.get_appearance_mode() == "Dark":
            bg = (29/255,30/255,30/255,255/255)
            # rgba(29,30,30,255)
            ft = "White"
            
        fig, ax = plt.subplots(figsize=(3.4, 1))  # Set figure size to 340x100 pixels


        ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)
        ax.text(0., 0.5, f'${latex_str}$', fontsize=20, ha='left', va='center', color=ft)
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)  # Set DPI to 100 to get 340x100 pixels
        buf.seek(0)
        img = Image.open(buf)
        return ImageTk.PhotoImage(img)
        
    # TODO
    def Calculate_event(self):
        """
        # Modify Output here(can call "update_output()")
        """
        
        print("Calculate_button click")
        self.Save_event()
        latex_str = evaluate_single("OutputFile.jpg")
        
        img = self.render_latex(latex_str)
        self.output_box.configure(state="normal")
        self.output_box.delete("0.0", "end")
        self.output_box._textbox.image_create("end", image=img)
        self.output_box.configure(state="disabled")
        print("Image rendered")
        # self.update_output("Output here")
        
        
        
    # end TODO
        
    def Draw_event(self):
        self.tool_mode = "pen"
        print("Draw_button click")
        
    def Erase_event(self):
        self.tool_mode = "eraser"
        print("Erase click")
    
    def Clean_event(self):
        self.canvas.delete("all")
        print("Clean click")
        
    def update_pen_size(self,size):
        self.pen_size = int(float(size))
        print("update pen")
    
    def update_eraser_size(self,size):
        self.eraser_size = int(float(size))
        print("update eraser")    


if __name__ == "__main__":
    app = App()
    app.mainloop()

