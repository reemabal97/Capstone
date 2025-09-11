# 🧠 Resume Classification (English + Arabic, LLM-based)

## 📌 Introduction
This project, **Resume Classification using Large Language Models (LLMs)**, aims to automatically classify resumes (CVs) into professional categories.  
The motivation comes from both the **need for automation in HR systems** and the **challenge of handling multilingual resumes** (English & Arabic).  

The dataset contains resumes in two versions:
- **English CV dataset** (~300 samples)  
- **Arabic CV dataset** (~300 samples)  

Both datasets were preprocessed with cleaning, text normalization, and feature extraction.

A total of 2 pipelines were implemented and compared:
- **English pipeline:** DistilBERT (fine-tuned with HuggingFace Transformers)  
- **Arabic pipeline:** AraBERT-mini (fine-tuned, with reshaping + bidi for visualization)  

Additionally:
- WordCloud visualizations (English & Arabic, with full font support for Arabic)  
- Streamlit web apps for interactive classification (`app_en.py`, `app_ar.py`)  
- Support for PDF, TXT, and DOCX file uploads  

---

## 📂 Code & Training Files
You can find the **training and evaluation codes** in the Python files included in this repository:

- `Train_en.py` → English resume classifier training  
- `Train_ar.py` → Arabic resume classifier training  
- `app_en.py` → Streamlit web app for English classification  
- `app_ar.py` → Streamlit web app for Arabic classification  

---

## ✅ Conclusion
This project successfully built **LLM-based systems for multilingual resume classification**.  

- The **English DistilBERT model** achieved solid accuracy on the evaluation set.  
- The **Arabic AraBERT-mini model** proved effective with small datasets when combined with preprocessing.  
- WordCloud visualizations were generated for both languages, with correct Arabic rendering thanks to custom font embedding.  

### Key findings:
- **Transfer learning** (DistilBERT, AraBERT) outperforms traditional ML baselines like Logistic Regression.  
- **Preprocessing** (reshaping, bidi, and font embedding) is essential for Arabic NLP.  
- Streamlit apps make the project easy to demo and extend.  

### Future extensions:
- Collecting a **larger and more diverse dataset**.  
- Improving accuracy with **data augmentation** and **hyperparameter tuning**.  
- Deploying the models as a **cloud-based web service** for HR automation.  
