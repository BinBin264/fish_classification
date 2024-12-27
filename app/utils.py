from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input

# Hàm tiền xử lý ảnh
def process_single_image(image_path, target_size=(224, 224)):
    # Đọc ảnh từ đường dẫn
    img = image.load_img(image_path, target_size=target_size)
    
    # Chuyển ảnh thành mảng NumPy
    img_array = image.img_to_array(img)
    
    # Thêm chiều batch (vì mô hình yêu cầu ảnh đầu vào có dạng [batch_size, height, width, channels])
    img_array = np.expand_dims(img_array, axis=0)
    
    # Áp dụng preprocess_input để chuẩn hóa ảnh theo yêu cầu của mô hình
    img_array = preprocess_input(img_array)
    
    return img_array


def process_label(pred_class):
    predicted_class = np.argmax(pred_class, axis=1)  # Chọn lớp có xác suất cao nhất
    # Giả sử bạn có một LabelEncoder đã được huấn luyện
    species = ['Anthias anthias', 'Atherinomorus lacunosus', 'Belone belone', 'Boops boops', 'Chlorophthalmus agassizi',
                'Coris julis', 'Dasyatis centroura', 'Epinephelus caninus', 'Gobius niger', 'Mugil cephalus', 'Phycis phycis',
                'Polyprion americanus', 'Pseudocaranx dentex', 'Rhinobatos cemiculus', 'Scomber japonicus', 'Solea solea',
                'Squalus acanthias', 'Tetrapturus belone', 'Trachinus draco', 'Trigloporus lastoviza']
    label_encoder = LabelEncoder()
    label_encoder.fit(species)
    # Chuyển chỉ số lớp thành nhãn loài cá
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label