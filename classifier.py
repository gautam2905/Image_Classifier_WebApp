from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np
from ultralytics import YOLO

# model = models.load_model(r"D:\ML Codes\ml_gui_app-main\ml_gui_app-main\finishedProject\baseline_mariya.keras")
model = YOLO("yolov8n-cls.pt")

def predict_image(model, path_to_img):
    results = model(path_to_img)
    name_list = model.names
    prob = results[0].probs
    top_prob = prob.top1
    top_prob1 = prob.top1conf.item()
    top_pred =  name_list[top_prob]

    return top_prob1, top_pred
    
content = ""
img_path = r"placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.jpg|>
select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred
        state.img_path = var_val
    #print(var_name, var_val)


app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)