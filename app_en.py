# app_en.py
import os, io, json, time, base64
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PyPDF2 import PdfReader
from docx import Document
import matplotlib.pyplot as plt

# ==================== THEME / STYLE ====================
st.set_page_config(page_title="Resume Classifier · EN (LLM)", page_icon="🧠", layout="wide")
PRIMARY = "#5B8DEF"
ACCENT  = "#22C55E"
DANGER  = "#EF4444"
BG      = "#0f172a"  # slate-900
CARD    = "#111827"  # slate-800
TEXT    = "#E5E7EB"  # slate-200

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
</style>
""", unsafe_allow_html=True)

# ==================== CONFIG ====================
DEFAULT_MODEL_DIR = "llm_model_en"
DEFAULT_TOPK = 5
DEFAULT_MAXLEN = 256  # طابِق سكربت التدريب لديك
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ==================== HELPERS ====================
@st.cache_resource(show_spinner=True)
def load_artifacts(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    labels_path = os.path.join(model_dir, "labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    mdl.eval()
    return tok, mdl, id2label

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def read_txt(file) -> str:
    content = file.read()
    try:
        return content.decode("utf-8")
    except Exception:
        return content.decode("latin-1", errors="ignore")

def read_pdf(file) -> str:
    # يدعم PDF كما طلبت
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()

def read_docx(file) -> str:
    # يدعم Word (DOCX) كما طلبت
    file_bytes = io.BytesIO(file.read())
    doc = Document(file_bytes)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def load_file_text(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):   return read_pdf(uploaded)
    if name.endswith(".txt"):   return read_txt(uploaded)
    if name.endswith(".docx"):  return read_docx(uploaded)
    raise ValueError("Unsupported file type. Please upload PDF, TXT, or DOCX.")

def predict_text(text: str, tok, mdl, id2label, device, max_len=DEFAULT_MAXLEN, top_k=DEFAULT_TOPK):
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        k = min(top_k, probs.shape[-1])
        conf, idx = torch.topk(probs, k=k)
    items = [(id2label[i.item()], float(c)) for i, c in zip(idx, conf)]
    return items

def meter(value, label, color=PRIMARY):
    return st.progress(0, text=label) if False else st.markdown(
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

def df_topk(items):
    return pd.DataFrame([{"label": lbl, "probability": round(p, 6)} for lbl, p in items])

# ==================== HEADER ====================
st.markdown(
    f"""
<div class="glass" style="padding:24px; margin-bottom:14px;">
  <div style="display:flex; gap:16px; align-items:center;">
    <div class="pill primary">LLM</div>
    <div class="pill">PDF</div>
    <div class="pill">TXT</div>
    <div class="pill">DOCX</div>
    <div class="pill accent">CSV Batch</div>
  </div>
  <h1 style="margin:6px 0 0 0;">🧠 Resume Classifier — English</h1>
  <div class="small-muted">Fine-tuned HuggingFace model · upload <b>PDF / TXT / DOCX</b> or paste text · batch CSV supported</div>
  <div class="small-muted">By : Reema Balharith</div>
</div>
""",
    unsafe_allow_html=True
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_dir = st.text_input("Model directory", value=DEFAULT_MODEL_DIR,
                              help="Folder containing HF model + tokenizer + labels.json")
    max_len = st.slider("Max tokens", 64, 512, DEFAULT_MAXLEN, step=32)
    top_k   = st.slider("Top-k", 1, 10, DEFAULT_TOPK)
    show_bars = st.checkbox("Show probability bars", value=True)
    device = pick_device()
    st.success(f"Device: {device}")

# -------- load model --------
try:
    tok, mdl, id2label = load_artifacts(model_dir)
    mdl.to(device)
except Exception as e:
    st.error(f"Failed to load model from **{model_dir}**\n\n{e}")
    st.stop()

# ==================== TABS ====================
t1, t2, t3 = st.tabs(["🔮 Predict", "📦 Batch (CSV)", "📊 EDA"])

# -------- PREDICT TAB --------
with t1:
    colA, colB = st.columns([1,1])

    with colA:
        st.subheader("Paste text")
        txt = st.text_area("Resume text (EN)", height=220, placeholder="Paste resume text here…")
        btn_txt = st.button("Predict from text", type="primary", use_container_width=True)

    with colB:
        st.subheader("Upload file (PDF / TXT / DOCX)")
        file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"], accept_multiple_files=False)
        btn_file = st.button("Predict from file", use_container_width=True)

    def run_predict(raw_text: str):
        if not raw_text.strip():
            st.warning("Please provide text or upload a file.")
            return
        with st.spinner("Running prediction…"):
            items = predict_text(raw_text, tok, mdl, id2label, device, max_len=max_len, top_k=top_k)
        pred, conf = items[0]
        st.markdown(f"### ✅ Prediction: **{pred}**  ·  Confidence: **{conf:.1%}**")
        if show_bars:
            for i, (lbl, p) in enumerate(items):
                meter(p*100, lbl, color=ACCENT if i==0 else PRIMARY)
        st.markdown("#### Top-k probabilities")
        st.dataframe(df_topk(items), use_container_width=True)

    if btn_txt:
        run_predict(txt)

    if btn_file:
        if file is None:
            st.warning("Upload a file first.")
        else:
            try:
                raw_text = load_file_text(file)
                st.caption(f"Loaded {len(raw_text)} characters from **{file.name}**")
                run_predict(raw_text)
            except Exception as e:
                st.error(f"Failed to read file: {e}")

# -------- BATCH TAB --------
with t2:
    st.subheader("Batch prediction from CSV")
    st.caption("CSV must contain a column named **text**.")
    csv = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="csvup")
    if csv is not None:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
        if "text" not in df.columns:
            st.error("CSV must include a column named `text`.")
        else:
            df["text"] = df["text"].fillna("").astype(str)
            preds, topk_json = [], []
            prog = st.progress(0, text="Predicting…")
            t0 = time.time()
            for i, t in enumerate(df["text"].tolist(), start=1):
                items = predict_text(t, tok, mdl, id2label, device, max_len=max_len, top_k=top_k)
                preds.append(items[0][0])
                topk_json.append(json.dumps([{"label":l,"prob":float(p)} for l,p in items], ensure_ascii=False))
                prog.progress(i/len(df))
            dt = time.time() - t0

            out = df.copy()
            out["prediction"] = preds
            out["topk_probs"] = topk_json

            st.success(f"Done: {len(df)} rows · {dt:.1f}s")
            st.dataframe(out.head(25), use_container_width=True)

            st.download_button(
                "⬇️ Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions_en.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------- EDA TAB --------
with t3:
    st.subheader("Quick EDA (by predictions)")
    st.caption("Paste or upload multiple resumes to see label distribution by model predictions.")
    files = st.file_uploader("Upload multiple files (PDF/TXT/DOCX)", type=["pdf","txt","docx"], accept_multiple_files=True, key="edafiles")
    bulk_txt = st.text_area("Or paste multiple resumes separated by ---", height=160, key="edatxt",
                            placeholder="Resume 1 ...\n---\nResume 2 ...\n---\nResume 3 ...")
    go = st.button("Run EDA", type="primary")

    if go:
        texts = []
        if bulk_txt.strip():
            parts = [p.strip() for p in bulk_txt.split("---") if p.strip()]
            texts.extend(parts)
        for f in files or []:
            try:
                texts.append(load_file_text(f))
            except Exception as e:
                st.warning(f"Skip {getattr(f,'name','file')}: {e}")

        if not texts:
            st.warning("Provide some resumes first (text or files).")
        else:
            labels_pred = []
            for t in texts:
                items = predict_text(t, tok, mdl, id2label, device, max_len=max_len, top_k=top_k)
                labels_pred.append(items[0][0])

            dist = pd.Series(labels_pred).value_counts().sort_values(ascending=False)
            st.markdown("#### Distribution (by predicted label)")
            fig, ax = plt.subplots(figsize=(8,4))
            dist.plot(kind="bar", ax=ax)
            ax.set_xlabel("Label"); ax.set_ylabel("Count"); ax.set_title("Predicted label counts")
            st.pyplot(fig, clear_figure=True)

            st.markdown("#### Table")
            st.dataframe(dist.reset_index().rename(columns={"index":"label", 0:"count"}), use_container_width=True)

# ==================== FOOTER ====================
st.markdown(
    f"""
<div class="small-muted" style="margin-top:24px; text-align:center;">
  Using model artifacts from <b>{DEFAULT_MODEL_DIR}</b>. You can change the path in the sidebar.<br/>
  Upload supports <b>PDF, TXT, DOCX</b> exactly as requested. Batch mode expects a CSV with a <code>text</code> column.
</div>
""",
    unsafe_allow_html=True
)
