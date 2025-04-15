# 🧴 Detect Words On Receipts

![EasyOCR](https://github.com/JaidedAI/EasyOCR#supported-languages)  
![Tesseract](https://github.com/tesseract-ocr/tesseract)  

## 📌 Project Overview

This project tackles the problem of **Detect and Classify words on daily receipts** using deep learning. We utilize a **DistilBert** model and fine-tune it on a curated dataset of receipts to predict various types of words.

Achieved a validation accuracy of **90%**.

> 📍 Kaggle notebook: [View here](https://www.kaggle.com/code/tantranduc/detect-words-on-receipt)  
> 📍 Reference: [EasyOCR](https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/)
> 📍 Reference: [Tesseract](https://pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)

---

## 🚀 Tech Stack

- Python
- TensorFlow / Keras
- DistilBert, Tesseract, EasyOCR
- NumPy, Pandas, Matplotlib
- Scikit-learn

---

## 🧠 Model Approach

1. **Data Preprocessing**
   - Extract words on receipts
   - labeling 

2. **Model**:
   - Used `DistilBert` model
   - Classify words based on label

3. **Evaluation**:
   - Accuracy: **90%**
   - Confusion matrix, precision/recall for each class

---

## 📊 Results

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | 90%       |

---

## 🧰 How to Run

```bash
mkdir yourfolder && cd yourfolder
git clone https://github.com/tantran1011/detect_words_on_receip.git

# Open the notebook
jupyter notebook detect-words-on-receip.ipynb
