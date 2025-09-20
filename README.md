# 🧠 Multilingual Resume Classification (English + Arabic) using LLMs  

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLP](https://img.shields.io/badge/NLP-LLM-green)

---

## 📌 Introduction  
This project, **Resume Classification using Large Language Models (LLMs)**, aims to automatically classify resumes (CVs) into professional categories.  
The motivation comes from both the **need for automation in HR systems** and the **challenge of handling multilingual resumes** (English & Arabic).  

The dataset contains resumes in two versions:
- **English CV dataset** (~6800 samples)  
- **Arabic CV dataset** (~6800 samples)  

Both datasets were preprocessed with cleaning, text normalization, and feature extraction.  

---

## 🚀 Features  
- Supports **English & Arabic resumes** 📝  
- Fine-tuned **DistilBERT** (English) & **AraBERT-mini** (Arabic)  
- **Interactive Streamlit app** for real-time classification  
- Handles **PDF, DOCX, TXT** uploads  
- **WordCloud visualizations** (with Arabic reshaping & bidi)  
- Fully documented training pipeline (Jupyter Notebook)  

---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/reemabal97/Resume-Classifier.git
cd Resume-Classifier
