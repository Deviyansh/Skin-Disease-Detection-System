import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import io

MAX_LEN = 20

def preprocess_image(file_bytes):
    img = image.load_img(io.BytesIO(file_bytes), target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_text(symptoms):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([symptoms])
    sequence = tokenizer.texts_to_sequences([symptoms])
    return pad_sequences(sequence, maxlen=MAX_LEN)


def get_disease_info(disease_name):
    with open("disease_info.txt", "r") as f:
        content = f.read()
        diseases = content.split("\n\n")

        for disease in diseases:
            lines = disease.strip().split("\n")
            if lines and lines[0].strip().rstrip(":") == disease_name:
                return {
                    "reason": lines[1].split("reason: ")[1],
                    "precaution": lines[2].split("precaution: ")[1],
                    "deadly": lines[3].split("deadly: ")[1],
                    "contagious": lines[4].split("contagious: ")[1],
                    "symptoms": lines[5].split("symptoms: ")[1],
                    "cure": lines[6].split("cure: ")[1],
                }
    return None
