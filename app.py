# app.py
import streamlit as st
import joblib

# جرّب pypdf أولًا (المستحسن)، واذا مو متوفرة ارجع إلى PyPDF2
try:
    from pypdf import PdfReader  # pip install pypdf
    USE_PYPDF = True
except ImportError:
    import PyPDF2  # pip install PyPDF2
    USE_PYPDF = False

st.set_page_config(page_title="Resume Classifier", page_icon="📄")

st.title("📄 Resume Classifier (PDF)")
st.write("ارفع سيرة ذاتية PDF وبنـصنّف المجال تلقائيًا 🤖")

# ===== تحميل النموذج والـ vectorizer مرة واحدة (Caching) =====
@st.cache_resource
def load_artifacts():
    model = joblib.load('cv_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# ===== وظائف مساعدة =====
def extract_text_from_pdf(uploaded_file) -> str:
    """
    تحاول تقرأ الـPDF وتستخرج النص.
    - تستخدم pypdf إن وجد، وإلا PyPDF2.
    - تتجاوز مشاكل الـstrict إذا ممكن.
    """
    if USE_PYPDF:
        # UploadedFile من ستريملت قابل للقراءة مباشرة
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        try:
            reader = PdfReader(uploaded_file, strict=False)
        except TypeError:
            # لإصدارات قديمة لا تدعم strict
            reader = PdfReader(uploaded_file)

        text = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text.append(page_text)
        return "\n".join(text).strip()
    else:
        # PyPDF2 احتياط
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text.append(page_text)
        return "\n".join(text).strip()

def classify_text(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
    return pred, proba

# ===== واجهة المستخدم =====
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    try:
        pdf_text = extract_text_from_pdf(uploaded_file)

        with st.expander("📃 عرض النص المستخرج"):
            st.write(pdf_text if pdf_text else "_لم يتم استخراج نص من الملف._")

        if st.button("Classify Resume"):
            if not pdf_text or len(pdf_text) < 20:
                st.warning("مافي نص كافي في الـPDF (قد يكون ممسوح/صور). جرّب PDF آخر أو نسخة نصية.")
            else:
                prediction, proba = classify_text(pdf_text)
                st.success(f"🔍 Predicted Resume Category: **{prediction}**")
                if proba is not None:
                    # أعرض أعلى 5 احتمالات لو تحبين
                    import numpy as np
                    classes = getattr(model, "classes_", None)
                    if classes is not None:
                        top_idx = np.argsort(proba)[::-1][:5]
                        st.subheader("Top probabilities")
                        for i in top_idx:
                            st.write(f"- {classes[i]}: {proba[i]:.3f}")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.info("نصائح: تأكدي من أن الملف PDF نصّي وليس ممسوح (صور). لو ممسوح، تحتاجين OCR.")
