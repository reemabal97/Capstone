
# app.py — Unified Streamlit (EN + AR), PDF/TXT/DOCX, Batch CSV, Styled UI

import os, io, re, json, time
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# ---------- optional readers ----------
try:
    from pypdf import PdfReader as PYPDFReader
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False

try:
    import pdfplumber
    _HAS_PLP = True
except Exception:
    _HAS_PLP = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

from docx import Document

# ==================== THEME / STYLE ====================
st.set_page_config(page_title="Resume Classifier · EN + AR", page_icon="🧠", layout="wide")
PRIMARY = "#5B8DEF"
ACCENT  = "#22C55E"
DANGER  = "#EF4444"
BG      = "#0f172a"
CARD    = "#111827"
TEXT    = "#E5E7EB"

st.markdown(f"""
<style>
  .stApp {{
    background: radial-gradient(1200px circle at 10% 0%, {BG} 0%, #030712 55%);
    color: {TEXT};
  }}
  .glass {{
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
  }}
  .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
           border:1px solid rgba(255,255,255,.15); margin-right:6px; }}
  .primary {{ color:white; background:{PRIMARY}; border:none; }}
  .accent {{ color:black; background:{ACCENT}; border:none; }}
  h1, h2, h3, h4 {{ color: #F8FAFC; }}
  .small-muted {{ color:#94A3B8; font-size:13px; }}
  .rtl * {{ direction: rtl; text-align: right; }}
</style>
""", unsafe_allow_html=True)

# ==================== CONFIG ====================
EN_MODEL_DIR = st.secrets.get("EN_REPO", "llm_model_en")
AR_MODEL_DIR = st.secrets.get("AR_REPO", "llm_model_ar")
DEFAULT_TOPK = 5
DEFAULT_MAXLEN = 256
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# ==================== HELPERS ====================
@st.cache_resource(show_spinner=True)
def load_artifacts(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    labels_path = os.path.join(model_dir, "labels.json")
    id2label = None
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "labels" in data:
            labels = data["labels"]
            id2label = {i: str(lbl) for i, lbl in enumerate(labels)}
        elif isinstance(data, dict) and "id2label" in data:
            id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        else:
            try:
                id2label = {int(k): str(v) for k, v in data.items()}
            except Exception:
                pass
    if id2label is None and hasattr(mdl.config, "id2label"):
        id2label = {int(k): str(v) for k, v in mdl.config.id2label.items()}
    if id2label is None:
        raise FileNotFoundError("labels.json not found or invalid, and no id2label in model config.")
    mdl.config.id2label = id2label
    mdl.config.label2id = {v: k for k, v in id2label.items()}
    mdl.eval()
    return tok, mdl, id2label

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def read_txt(file) -> str:
    b = file.read()
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def read_docx(file) -> str:
    bio = io.BytesIO(file.read())
    doc = Document(bio)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def pdf_text_english(file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception:
        return ""

def _pdf_text_with_pypdf(file) -> str:
    reader = PYPDFReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def _pdf_text_with_pdfplumber(file) -> str:
    if hasattr(file, "read"):
        b = file.read()
        if hasattr(file, "seek"): file.seek(0)
    else:
        b = file
    txt_pages=[]
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            txt_pages.append(t)
    out = "\n".join(txt_pages)
    return re.sub(r"\\s+", " ", out).strip()

def _pdf_text_with_pdfminer(file) -> str:
    if hasattr(file, "read"):
        b = file.read()
        bio = io.BytesIO(b)
        txt = pdfminer_extract(bio)
        if hasattr(file, "seek"): file.seek(0)
    else:
        txt = pdfminer_extract(file)
    return txt

def pdf_text_arabic(file) -> str:
    txt = ""
    try:
        if _HAS_PYPDF:
            txt = _pdf_text_with_pypdf(file)
    except Exception:
        txt = ""
    def bad(s: str) -> bool:
        s = (s or "").strip()
        return (len(s) < 120) or ("ﻼ" in s) or (s.count("\\u200f") > 50)
    if bad(txt) and _HAS_PLP:
        try:
            if hasattr(file, "seek"): file.seek(0)
            t2 = _pdf_text_with_pdfplumber(file)
            if len(t2) > len(txt): txt = t2
        except Exception:
            pass
    if bad(txt) and _HAS_PDFMINER:
        try:
            if hasattr(file, "seek"): file.seek(0)
            t3 = _pdf_text_with_pdfminer(file)
            if len(t3) > len(txt): txt = t3
        except Exception:
            pass
    return re.sub(r"\\s+", " ", txt or "").strip()

def load_file_text(uploaded, lang: str) -> str:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        if hasattr(uploaded, "seek"): uploaded.seek(0)
        return pdf_text_arabic(uploaded) if lang=="Arabic" else pdf_text_english(uploaded)
    if name.endswith(".txt"):   return read_txt(uploaded)
    if name.endswith(".docx"):  return read_docx(uploaded)
    raise ValueError("Unsupported file type. Upload PDF / TXT / DOCX.")

def normalize_ar(text: str) -> str:
    _map = str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ؤ":"و","ئ":"ي","ة":"ه"})
    text = re.sub(r"[\\u0617-\\u061A\\u064B-\\u0652]", "", text or "")
    text = (text or "").translate(_map)
    text = re.sub(r"[^\\u0600-\\u06FF0-9a-zA-Z\\s%+@\\-\\.]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def predict_text(text: str, tok, mdl, id2label, device, max_len=DEFAULT_MAXLEN, top_k=DEFAULT_TOPK, lang="English"):
    txt = normalize_ar(text) if lang=="Arabic" else (text or "")
    enc = tok(txt, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        k = min(top_k, probs.shape[-1])
        conf, idx = torch.topk(probs, k=k)
    items = [(id2label[i.item()], float(c)) for i, c in zip(idx, conf)]
    return items

def df_topk(items):
    return pd.DataFrame([{"label": lbl, "probability": round(p, 6)} for lbl, p in items])

def meter(value, label, color=PRIMARY):
    st.markdown(
        f"""<div class="glass" style="background:{CARD}; margin:6px 0;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-weight:600;">{label}</div>
                <div style="font-weight:700;">{value:.1f}%</div>
              </div>
              <div style="height:10px; background:#1f2937; border-radius:999px; overflow:hidden; margin-top:6px;">
                <div style="height:10px; width:{value}%; background:{color};"></div>
              </div>
            </div>""",
        unsafe_allow_html=True
    )

# ==================== HEADER ====================
st.markdown(
    f"""
<div class="glass" style="padding:24px; margin-bottom:14px;">
  <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
    <div class="pill primary">LLM</div>
    <div class="pill">PDF</div>
    <div class="pill">TXT</div>
    <div class="pill">DOCX</div>
    <div class="pill accent">CSV Batch</div>
  </div>
  <h1 style="margin:6px 0 0 0;">🧠 Resume Classifier — English & Arabic</h1>
  <div class="small-muted">Upload <b>PDF / TXT / DOCX</b> or paste text · batch CSV supported · language-aware pipeline</div>
  <div class="small-muted">By : Reema Balharith</div>
</div>
""",
    unsafe_allow_html=True
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    lang = st.radio("Language / اللغة", ["English","Arabic"], horizontal=True)
    st.markdown('<div class="rtl"></div>' if lang=="Arabic" else "", unsafe_allow_html=True)

    # -------- Sticky Model IDs --------
    # لو ما فيه قيمة محفوظة، نخليها من Secrets أو الديفولت
    if "en_repo" not in st.session_state:
        st.session_state["en_repo"] = EN_MODEL_DIR
    if "ar_repo" not in st.session_state:
        st.session_state["ar_repo"] = AR_MODEL_DIR

    # نربط text_input مع session_state مباشرة
    en_model_dir = st.text_input(
        "EN model (HF repo id)", 
        value=st.session_state["en_repo"], 
        key="en_repo"
    )
    ar_model_dir = st.text_input(
        "AR model (HF repo id)", 
        value=st.session_state["ar_repo"], 
        key="ar_repo"
    )

    max_len = st.slider("Max tokens", 64, 512, DEFAULT_MAXLEN, step=32)
    top_k   = st.slider("Top-k", 1, 10, DEFAULT_TOPK)
    device = torch.device("cpu")  # ستريملت كلاود = CPU
    st.success(f"Device: {device}")

# ==================== TABS ====================
t1, t2, t3 = st.tabs(["🔮 Predict", "📦 Batch (CSV)", "📊 EDA"])

# -------- PREDICT TAB --------
with t1:
    cols = st.columns([1,1])
    with cols[0]:
        st.subheader("Paste text" if lang=="English" else "لصق نص")
        placeholder = "Paste resume text here…" if lang=="English" else "الصقي نص السيرة الذاتية هنا…"
        txt = st.text_area(placeholder, height=220)
        btn_txt = st.button("Predict from text" if lang=="English" else "تنبؤ من النص",
                            type="primary", use_container_width=True)

    with cols[1]:
        st.subheader("Upload file (PDF / TXT / DOCX)" if lang=="English" else "رفع ملف (PDF / TXT / DOCX)")
        file = st.file_uploader("Choose a file" if lang=="English" else "اختاري ملفًا",
                                type=["pdf","txt","docx"], accept_multiple_files=False)
        btn_file = st.button("Predict from file" if lang=="English" else "تنبؤ من الملف",
                             use_container_width=True)

    def run_predict(raw_text: str):
        if not raw_text.strip():
            st.warning("Please provide text or upload a file." if lang=="English" else "فضلاً ادخلي نصًا أو ارفعي ملفًا.")
            return
        with st.spinner("Running prediction…" if lang=="English" else "جارٍ التنبؤ…"):
            items = predict_text(raw_text, tok, mdl, id2label, device, max_len=max_len, top_k=top_k, lang=lang)
        pred, conf = items[0]
        st.markdown(
            f"### ✅ Prediction: **{pred}**  ·  Confidence: **{conf:.1%}**"
            if lang=="English" else
            f"### ✅ الفئة المتوقعة: **{pred}**  ·  الوثوقية: **{conf:.1%}**"
        )
        if show_bars:
            for i,(lbl,p) in enumerate(items):
                meter(p*100, lbl, color=ACCENT if i==0 else PRIMARY)
        st.markdown("#### Top-k probabilities" if lang=="English" else "#### أفضل النتائج (Top-k)")
        st.dataframe(df_topk(items), use_container_width=True)

    if btn_txt:
        run_predict(txt)
    if btn_file:
        if file is None:
            st.warning("Upload a file first." if lang=="English" else "فضلاً اختاري ملفًا أولاً.")
        else:
            try:
                raw_text = load_file_text(file, lang)
                st.caption(
                    f"Loaded {len(raw_text)} characters from **{file.name}**"
                    if lang=="English" else f"تم تحميل **{file.name}** — عدد الأحرف: {len(raw_text)}"
                )
                run_predict(raw_text)
            except Exception as e:
                st.error(f"Failed to read file: {e}" if lang=="English" else f"تعذّر قراءة الملف: {e}")

# -------- BATCH TAB --------
with t2:
    st.subheader("Batch prediction from CSV" if lang=="English" else "تنبؤ دفعي من CSV")
    st.caption(
        "CSV must contain a column named **text**."
        if lang=="English" else "يجب أن يحتوي CSV على عمود اسمه **text** (نص السيرة)."
    )
    csv = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="csvup_any")
    if csv is not None:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}" if lang=="English" else f"تعذّر قراءة CSV: {e}")
            st.stop()
        if "text" not in df.columns:
            st.error("CSV must include a column named `text`." if lang=="English" else "يجب أن يحتوي CSV على عمود باسم `text`.")
        else:
            df["text"] = df["text"].fillna("").astype(str)
            preds, topk_json = [], []
            prog = st.progress(0, text="Predicting…" if lang=="English" else "تشغيل التنبؤات…")
            t0 = time.time()
            for i, t in enumerate(df["text"].tolist(), start=1):
                items = predict_text(t,
                                     tok, mdl, id2label, device,
                                     max_len=max_len, top_k=top_k, lang=lang)
                preds.append(items[0][0])
                topk_json.append(json.dumps([{"label":l,"prob":float(p)} for l,p in items], ensure_ascii=False))
                prog.progress(i/len(df))
            dt = time.time() - t0

            out = df.copy()
            out["prediction"] = preds
            out["topk_probs"] = topk_json

            st.success(
                f"Done: {len(df)} rows · {dt:.1f}s"
                if lang=="English" else f"تم: {len(df)} صفًا · {dt:.1f}ث"
            )
            st.dataframe(out.head(25), use_container_width=True)

            st.download_button(
                "⬇️ Download predictions" if lang=="English" else "⬇️ تنزيل النتائج",
                out.to_csv(index=False).encode("utf-8"),
                file_name=("predictions_en.csv" if lang=="English" else "predictions_ar.csv"),
                mime="text/csv",
                use_container_width=True
            )

# -------- EDA TAB --------
with t3:
    st.subheader("Quick EDA (by predictions)" if lang=="English" else "استكشاف سريع (حسب تنبؤات النموذج)")
    st.caption(
        "Paste or upload multiple resumes to see label distribution by model predictions."
        if lang=="English" else "ارفعي عدة ملفات أو الصقي نصوصًا مفصولة بـ --- لعرض توزيع الفئات المتوقعة."
    )
    files = st.file_uploader("Upload multiple files (PDF/TXT/DOCX)" if lang=="English" else "ارفعي عدة ملفات (PDF/TXT/DOCX)",
                             type=["pdf","txt","docx"], accept_multiple_files=True, key="edafiles_any")
    bulk_txt = st.text_area(
        "Or paste multiple resumes separated by ---" if lang=="English" else "أو الصقي عدة سير مفصولة بـ ---",
        height=160, key="edatxt_any"
    )
    go = st.button("Run EDA" if lang=="English" else "تشغيل الاستكشاف", type="primary")

    if go:
        texts=[]
        if bulk_txt.strip():
            parts=[p.strip() for p in bulk_txt.split("---") if p.strip()]
            texts.extend(parts)
        for f in files or []:
            try:
                texts.append(load_file_text(f, lang))
            except Exception as e:
                st.warning(f"Skip {getattr(f,'name','file')}: {e}" if lang=="English" else f"تخطي {getattr(f,'name','file')}: {e}")

        if not texts:
            st.warning("Provide some resumes first (text or files)." if lang=="English" else "فضلاً ادخلي نصوصًا أو ارفعي ملفات أولاً.")
        else:
            labels_pred=[]
            for t in texts:
                items = predict_text(t, tok, mdl, id2label, device, max_len=DEFAULT_MAXLEN, top_k=DEFAULT_TOPK, lang=lang)
                labels_pred.append(items[0][0])

            dist = pd.Series(labels_pred).value_counts().sort_values(ascending=False)
            st.markdown("#### Distribution (by predicted label)" if lang=="English" else "#### توزيع الفئات المتوقعة")
            fig, ax = plt.subplots(figsize=(8,4))
            color = "#38bdf8" if lang=="Arabic" else "#22c55e"
            dist.plot(kind="bar", ax=ax, color=color)
            ax.set_xlabel("Label" if lang=="English" else "الفئة")
            ax.set_ylabel("Count" if lang=="English" else "العدد")
            ax.set_title("Predicted label counts" if lang=="English" else "توزيع التنبؤات")
            st.pyplot(fig, clear_figure=True)

            st.markdown("#### Table" if lang=="English" else "#### جدول")
            st.dataframe(
                dist.reset_index().rename(columns={"index":"label" if lang=="English" else "الفئة", 0:"count" if lang=="English" else "العدد"}),
                use_container_width=True
            )

# ==================== FOOTER ====================
st.markdown(
    f"""
<div class="small-muted" style="margin-top:24px; text-align:center;">
  Models: <b>{EN_MODEL_DIR}</b> (EN) · <b>{AR_MODEL_DIR}</b> (AR). You can change these in the sidebar.<br/>
  Upload supports <b>PDF, TXT, DOCX</b> · Batch mode expects CSV column <code>text</code>.
</div>
""",
    unsafe_allow_html=True
)
