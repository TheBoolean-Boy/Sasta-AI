import tkinter as tk
import customtkinter as ctk 


from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 


app = tk.Tk()
app.geometry("432x632")
app.title("Sasta AI") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(master=app, placeholder_text="C ", width=412,  height=40, border_width=2,corner_radius=10)
prompt.place(x=10, y=10)

text_var = tk.StringVar(value="image")

lmain = ctk.CTkLabel(master=app,textvariable=text_var, width=412, height=412,fg_color=("white", "gray75"),corner_radius=8)

lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-2"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

def generate(): 
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]


        image.save('Ge.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

trigger = ctk.CTkButton(master=app, width=120, height=40, border_width=0, corner_radius=8, text="CTkButton", command=generate)
trigger.configure(text="Kala Jadu kardo") 
trigger.place(x=206, y=60) 

app.mainloop()