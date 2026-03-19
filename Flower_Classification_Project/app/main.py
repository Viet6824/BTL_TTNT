import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import traceback

# 104 Flower Classes
FLOWER_CLASSES = [
    'anh thảo hồng', 'lan hài lá cứng', 'hoa chuông canterbury', 'hoa đậu thơm', 'cúc vạn thọ anh', 
    'hoa huệ hổ', 'lan hồ điệp', 'hoa thiên điểu', 'hoa ô đầu', 'hoa kế địa cầu', 'hoa mõm sói', 'hoa khoản đông', 
    'hoa thảo đường hoàng đế', 'kế ngọn giáo', 'diên vĩ vàng', 'hoa cầu vàng', 'cúc nhím tím', 'hoa ly peru', 
    'hoa cát cánh', 'hoa rum trắng khổng lồ', 'huệ tây đỏ', 'hoa gối cắm kim', 'hoa xuyên bối mẫu', 'hoa riềng đỏ', 
    'lan dạ hương nho', 'hoa anh túc đỏ', 'mào gà lông', 'long đởm không thân', 'atisô', 'cẩm chướng lùn', 
    'hoa cẩm chướng', 'hoa phlox', 'hoa tình yêu trong sương', 'cúc sao nhái', 'mùi tàu vùng núi cao', 'lan cattleya môi đỏ', 
    'hoa cúc cape', 'hoa cúc masterwort', 'hoa ngải tiên', 'hoa hồng lenten', 'đồng tiền barbeton', 'hoa thủy tiên', 'hoa lay ơn', 
    'hoa trạng nguyên', 'cúc eustoma xanh', 'hoa quế trúc', 'cúc vạn thọ', 'hoa mao lương', 'cúc oxeye', 'bồ công anh', 
    'hoa dạ yến thảo', 'hoa păng xê dại', 'hoa anh thảo', 'hoa hướng dương', 'hoa phong lữ', 'thược dược bishop', 'hoa gaura', 'phong lữ thảo', 
    'thược dược cam', 'thược dược vàng hồng', 'gừng hương cautleya', 'thu mẫu đơn nhật bản', 'cúc susan mắt đen', 'cúc bạc silverbush', 
    'anh túc california', 'cúc châu phi', 'hoa nghệ tây mùa xuân', 'diên vĩ có râu', 'hoa phong quỳ', 'hoa anh túc cây', 'cúc huân chương', 
    'hoa đỗ quyên', 'hoa súng', 'hoa hồng', 'cà độc dược', 'hoa bìm bìm', 'hoa lạc tiên', 'hoa sen', 'lan cóc', 
    'hoa hồng môn', 'hoa đại', 'hoa ông lão', 'hoa dâm bụt', 'hoa bồ câu', 'hoa sứ', 'hoa cẩm quỳ cây', 'hoa mộc lan', 
    'hoa tiên ông', 'cải xoong', 'hoa chuối cảnh', 'lan huệ', 'cúc nữ hoàng', 'rêu bóng', 'hoa mao địa hoàng', 'hoa giấy', 
    'hoa trà', 'hoa cẩm quỳ', 'chiều tím', 'hoa dứa cảnh', 'cúc gaillardia', 'hoa đăng tiêu', 'rẻ quạt'
]

app = FastAPI(title="Petal Predictor API")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=templates_dir)

model = None
effnet_preprocess = None
USE_EFFNET = False

@app.on_event("startup")
def load_ai():
    global model, USE_EFFNET, effnet_preprocess
    print("Starting AI Engine, please wait...")
    try:
        import sys
        
        # Load TF
        import tensorflow as tf
        from tensorflow.keras.applications.efficientnet import preprocess_input
        effnet_preprocess = preprocess_input
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        models_to_check = ['model_effnet_v2.h5', 'model_effnet.h5', 'model_v1.h5']
        
        for m in models_to_check:
            m_path = os.path.join(base_dir, m)
            if os.path.exists(m_path):
                print(f"Loading {m_path}...")
                model = tf.keras.models.load_model(m_path, compile=False)
                USE_EFFNET = ('effnet' in m)
                print("Successfully loaded AI Model!")
                break
        
        if model is None:
            print("WARNING: No .h5 file found. Running in Demo mode.")
    except Exception as e:
        print("ERROR LOADING TENSORFLOW/MODEL:", e)
        traceback.print_exc()

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    if USE_EFFNET and effnet_preprocess:
        img_array = effnet_preprocess(img_array)
    else:
        img_array = img_array / 255.0  # Normalize cho MobileNet
        
    img_array = np.expand_dims(img_array, axis=0) # shape (1, 224, 224, 3)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        if model is not None:
            img_tensor = preprocess_image(contents)
            
            # Predict
            predictions = model(img_tensor, training=False).numpy()[0]
            
            class_idx = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            
            if class_idx < len(FLOWER_CLASSES):
                class_name = FLOWER_CLASSES[class_idx]
            else:
                class_name = f"Class {class_idx}"
                
            return JSONResponse(content={"class_name": class_name, "confidence": confidence})
        else:
            # DEMO MODE
            import random
            return JSONResponse(content={
                "class_name": random.choice(FLOWER_CLASSES) + " (DEMO)",
                "confidence": 0.85 + random.random() * 0.14
            })
    except Exception as e:
        print("API ERROR:", e)
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
