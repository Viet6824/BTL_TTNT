import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from PIL import Image

def get_class():
    model_path = r'd:\BLT TTNT\Flower_Classification_Project\model_v1.h5'
    image_path = r'C:\Users\Administrator\.gemini\antigravity\brain\e2a61e7f-4585-4be5-be9a-5b952d7f6039\uploaded_media_1772473933043.png'

    model = tf.keras.models.load_model(model_path)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

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
    
    print(f"Index: {class_idx}")
    print(f"Predicted Class: {FLOWER_CLASSES[class_idx]}")
    print(f"Confidence: {confidence:.4f}")
    
    # Check top 3 predictions
    top_3 = np.argsort(predictions[0])[-3:][::-1]
    for i in top_3:
        print(f"Top: {FLOWER_CLASSES[i]} ({predictions[0][i]:.4f}) -> Index {i}")

if __name__ == '__main__':
    get_class()
