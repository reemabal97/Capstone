# app_ar.py — عربي + RTL + PDF/TXT/DOCX + Debug + Normalization
import pdfplumber
import os, io, json, time, re
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# ملاحظة: سنحاول pypdf أولًا، ولو فشل نرجع لـ pdfminer
try:
    from pypdf import PdfReader as PYPDFReader
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False
from docx import Document
import matplotlib.pyplot as plt

# ==================== إعداد الواجهة / الألوان ====================
st.set_page_config(page_title="مصنّف السير الذاتية (عربي)", page_icon="🧠", layout="wide")
PRIMARY = "#10B981"  # emerald
ACCENT  = "#60A5FA"  # blue
DANGER  = "#EF4444"
BG      = "#0f172a"  # slate-900
CARD    = "#111827"  # slate-800
TEXT    = "#E5E7EB"  # slate-200

st.markdown(f"""
<style>
  .stApp {{
    direction: rtl;
    text-align: right;
    background: radial-gradient(1200px circle at 90% 0%, {BG} 0%, #030712 55%);
    color: {TEXT};
    font-family: "Cairo","Tajawal",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans","Apple Color Emoji","Segoe UI Emoji";
  }}
  .glass {{
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
  }}
  .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
           border:1px solid rgba(255,255,255,.15); margin-left:6px; }}
  .primary {{ color:black; background:{PRIMARY}; border:none; }}
  .accent {{ color:black; background:{ACCENT}; border:none; }}
  h1, h2, h3, h4 {{ color:#F8FAFC; }}
  .small-muted {{ color:#94A3B8; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# ==================== الإعدادات ====================
DEFAULT_MODEL_DIR = "llm_model_ar"
DEFAULT_TOPK = 5
DEFAULT_MAXLEN = 256
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ==================== تطبيع عربي ====================
_ar_norm_map = str.maketrans({
    "أ":"ا","إ":"ا","آ":"ا","ى":"ي","ؤ":"و","ئ":"ي","ة":"ه",
})
_tashkeel = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
def normalize_ar(text: str) -> str:
    # إزالة التشكيل وتوحيد الحروف + تبسيط الرموز والمسافات
    text = _tashkeel.sub("", text)
    text = text.translate(_ar_norm_map)
    text = re.sub(r"[^\u0600-\u06FF0-9a-zA-Z\s%+@\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==================== تحميل النموذج والملصقات ====================
@st.cache_resource(show_spinner=True)
def load_artifacts(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    labels_path = os.path.join(model_dir, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"لم أجد labels.json داخل المجلد: {model_dir}")
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # يدعم شكل {"labels":[...]} أو {"id2label":{..},"label2id":{..}}
    if "labels" in data:
        labels = data["labels"]
        id2label = {i: lbl for i, lbl in enumerate(labels)}
    else:
        id2label = {int(k): v for k, v in data["id2label"].items()}
    # ثبتيها داخل الكونفق (مهم لتطابق المخرجات)
    mdl.config.id2label = id2label
    mdl.config.label2id = {v: k for k, v in id2label.items()}
    mdl.eval()
    return tok, mdl, id2label

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ==================== قرّاء الملفات ====================
def read_txt(file) -> str:
    content = file.read()
    try:
        return content.decode("utf-8")
    except Exception:
        return content.decode("latin-1", errors="ignore")

def _pdf_text_with_pypdf(file) -> str:
    reader = PYPDFReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def _pdf_text_with_pdfminer(file) -> str:
    # pdfminer يحتاج مسار أو BytesIO
    from pdfminer.high_level import extract_text
    if hasattr(file, "read"):
        b = file.read()
        bio = io.BytesIO(b)
        txt = extract_text(bio)
    else:
        txt = extract_text(file)
    return txt

def _pdf_text_with_pdfplumber(file) -> str:
    """
    استخراج نص PDF بدقة أفضل للعربي وبدون OCR.
    يقبل UploadedFile أو bytes.
    """
    # خذي البايتات من الملف المرفوع
    if hasattr(file, "read"):
        b = file.read()
        if hasattr(file, "seek"): 
            file.seek(0)  # نرجّع المؤشر لأننا قد نقرأه لاحقًا
    else:
        b = file  # bytes

    txt_pages = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            txt_pages.append(t)

    out = "\n".join(txt_pages)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def extract_text_ar(uploaded) -> str:
    """
    نحاول pypdf أولًا؛ لو سيّئ نجرب pdfplumber؛ لو ما نفع نجرب pdfminer.
    """
    txt = ""

    # 1) pypdf أولًا (لو متوفر)
    try:
        if _HAS_PYPDF:
            txt = _pdf_text_with_pypdf(uploaded)
        else:
            txt = ""
    except Exception:
        txt = ""

    # معيار بسيط للحكم على سوء النص
    def is_bad(s: str) -> bool:
        s = (s or "").strip()
        return (len(s) < 120) or ("ﻼ" in s) or (s.count("\u200f") > 50)

    # 2) لو سيّئ → pdfplumber
    if is_bad(txt):
        try:
            if hasattr(uploaded, "seek"): uploaded.seek(0)
            txt2 = _pdf_text_with_pdfplumber(uploaded)
            if len(txt2) > len(txt): 
                txt = txt2
        except Exception:
            pass

    # 3) لو ما زال سيّئ → pdfminer (الموجود عندك سابقًا)
    if is_bad(txt):
        try:
            if hasattr(uploaded, "seek"): uploaded.seek(0)
            txt3 = _pdf_text_with_pdfminer(uploaded)
            if len(txt3) > len(txt): 
                txt = txt3
        except Exception:
            pass

    return re.sub(r"\s+", " ", txt or "").strip()


def read_docx(file) -> str:
    file_bytes = io.BytesIO(file.read())
    doc = Document(file_bytes)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def load_file_text(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        # نحتاج إعادة المؤشر لأن الدوال قد تقرأه أكثر من مرّة
        if hasattr(uploaded, "seek"): uploaded.seek(0)
        return extract_text_ar(uploaded)
    if name.endswith(".txt"):   return read_txt(uploaded)
    if name.endswith(".docx"):  return read_docx(uploaded)
    raise ValueError("صيغة غير مدعومة. ارفعي PDF أو TXT أو DOCX.")

# ==================== التنبؤ ====================
def predict_text(raw_text: str, tok, mdl, id2label, device, max_len=DEFAULT_MAXLEN, top_k=DEFAULT_TOPK):
    # تطبيع عربي قبل التوكننة
    text = normalize_ar(raw_text)
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        k = min(top_k, probs.shape[-1])
        conf, idx = torch.topk(probs, k=k)
    items = [(id2label[i.item()], float(c)) for i, c in zip(idx, conf)]
    return items, text  # نرجع النص بعد التطبيع للأغراض التشخيصية


def bar_row(value, label, color=PRIMARY):
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

def df_topk(items):
    return pd.DataFrame([{"الفئة": lbl, "الاحتمال": round(p, 6)} for lbl, p in items])

# ==================== رأس الصفحة ====================
st.markdown(
    f"""
<div class="glass" style="padding:24px; margin-bottom:14px;">
  <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
    <div class="pill primary">نموذج LLM</div>
    <div class="pill">PDF</div>
    <div class="pill">TXT</div>
    <div class="pill">DOCX</div>
    <div class="pill accent">CSV دفعي</div>
  </div>
  <h1 style="margin:6px 0 0 0;">🧠 مصنّف السير الذاتية — عربي</h1>
  <div class="small-muted">ارفعي <b>PDF / TXT / DOCX</b> أو الصقي النص مباشرة. يدعم التنبؤ الدفعي عبر CSV.</div>
  <div class="small-muted">بواسطة: ريما بالحارث</div>
</div>
""",
    unsafe_allow_html=True
)

# ==================== الشريط الجانبي ====================
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    model_dir = st.text_input("مجلد النموذج", value=DEFAULT_MODEL_DIR,
                              help="المجلد الذي يحوي ملفات النموذج والمحوّل وملف labels.json")
    max_len = st.slider("أقصى طول (توكنز)", 64, 512, DEFAULT_MAXLEN, step=32)
    top_k   = st.slider("عدد النتائج (Top-k)", 1, 10, DEFAULT_TOPK)
    show_bars = st.checkbox("إظهار أشرطة الاحتمالات", value=True)
    device = pick_device()
    st.success(f"الجهاز المستخدم: {device}")

# تحميل النموذج
try:
    tok, mdl, id2label = load_artifacts(model_dir)
    mdl.to(device)
except Exception as e:
    st.error(f"تعذّر تحميل النموذج من **{model_dir}**\n\n{e}")
    st.stop()

# ==================== التبويبات ====================
t1, t2, t3 = st.tabs(["🔮 تنبؤ", "📦 دفعي (CSV)", "📊 استكشاف (EDA)"])

# -------- تبويب التنبؤ --------
with t1:
    colA, colB = st.columns([1,1])

    with colA:
        st.subheader("لصق نص")
        txt = st.text_area("الصقي نص السيرة الذاتية هنا:", height=220, placeholder="اكتبي أو الصقي النص…")
        btn_txt = st.button("تنبؤ من النص", type="primary", use_container_width=True)

    with colB:
        st.subheader("رفع ملف (PDF / TXT / DOCX)")
        file = st.file_uploader("اختاري ملفًا", type=["pdf", "txt", "docx"], accept_multiple_files=False)
        btn_file = st.button("تنبؤ من الملف", use_container_width=True)

    def run_predict_ar(raw_text: str, *, show_debug_block: bool = True):
        if not raw_text or not raw_text.strip():
            st.warning("فضلاً ادخلي نصًا أو ارفعي ملفًا.")
            return
        # نعرض مقتطف قبل وما بعد التطبيع
        if show_debug_block:
            with st.expander("🔍 مقتطف من النص قبل التصنيف (للتأكد من سلامته)", expanded=True):
                st.code((raw_text or "")[:500], language="text")
        with st.spinner("جارٍ التنبؤ…"):
            items, norm_text = predict_text(raw_text, tok, mdl, id2label, device, max_len=max_len, top_k=top_k)
        # تحذير لو النص قصير جدًا بعد الاستخراج
        if len((raw_text or "").strip()) < 120:
            st.warning("النص المستخرج قصير جدًا — قد تكون نتيجة PDF غير دقيقة. جرّبي رفع نسخة أخرى أو استخدام pdfminer.")
        pred, conf = items[0]
        st.markdown(f"### ✅ الفئة المتوقعة: **{pred}**  ·  الوثوقية: **{conf:.1%}**")
        if show_bars:
            for i, (lbl, p) in enumerate(items):
                bar_row(p*100, lbl, color=ACCENT if i==0 else PRIMARY)
        st.markdown("#### أفضل النتائج (Top-k)")
        st.dataframe(df_topk(items), use_container_width=True)

    if btn_txt:
        run_predict_ar(txt)

    if btn_file:
        if file is None:
            st.warning("فضلاً اختاري ملفًا أولاً.")
        else:
            try:
                raw_text = load_file_text(file)
                st.caption(f"تم تحميل **{file.name}** — عدد الأحرف: {len(raw_text)}")
                run_predict_ar(raw_text)
            except Exception as e:
                st.error(f"تعذّر قراءة الملف: {e}")

# -------- تبويب الدفعي --------
with t2:
    st.subheader("تنبؤ دفعي من CSV")
    st.caption("يجب أن يحتوي CSV على عمود اسمه **text** (نص السيرة).")
    csv = st.file_uploader("ارفع CSV", type=["csv"], accept_multiple_files=False, key="csvup_ar")
    if csv is not None:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            st.error(f"تعذّر قراءة CSV: {e}")
            st.stop()
        if "text" not in df.columns:
            st.error("يجب أن يحتوي CSV على عمود باسم `text`.")
        else:
            df["text"] = df["text"].fillna("").astype(str)
            preds, topk_json = [], []
            prog = st.progress(0, text="تشغيل التنبؤات…")
            t0 = time.time()
            for i, t in enumerate(df["text"].tolist(), start=1):
                items, _ = predict_text(t, tok, mdl, id2label, device, max_len=max_len, top_k=top_k)
                preds.append(items[0][0])
                topk_json.append(json.dumps([{"label":l,"prob":float(p)} for l,p in items], ensure_ascii=False))
                prog.progress(i/len(df))
            dt = time.time() - t0

            out = df.copy()
            out["prediction"] = preds
            out["topk_probs"] = topk_json

            st.success(f"تم: {len(df)} صفًا · {dt:.1f}ث")
            st.dataframe(out.head(25), use_container_width=True)

            st.download_button(
                "⬇️ تنزيل النتائج",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions_ar.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------- تبويب الاستكشاف --------
with t3:
    st.subheader("استكشاف سريع (حسب تنبؤات النموذج)")
    st.caption("ارفعي عدة ملفات أو الصقي عدّة نصوص مفصولة بـ --- لرؤية توزيع الفئات المتوقعة.")
    files = st.file_uploader("ارفعي عدة ملفات (PDF/TXT/DOCX)", type=["pdf","txt","docx"], accept_multiple_files=True, key="edafiles_ar")
    bulk_txt = st.text_area("أو الصقي عدة سير (افصلي بينها بـ ---)", height=160, key="edatxt_ar",
                            placeholder="سيرة 1 …\n---\nسيرة 2 …\n---\nسيرة 3 …")
    go = st.button("تشغيل الاستكشاف", type="primary")

    if go:
        texts = []
        if bulk_txt.strip():
            parts = [p.strip() for p in bulk_txt.split("---") if p.strip()]
            texts.extend(parts)
        for f in files or []:
            try:
                texts.append(load_file_text(f))
            except Exception as e:
                st.warning(f"تخطي {getattr(f,'name','file')}: {e}")

        if not texts:
            st.warning("فضلاً ادخلي نصوصًا أو ارفعي ملفات أولاً.")
        else:
            labels_pred = []
            for t in texts:
                items, _ = predict_text(t, tok, mdl, id2label, device, max_len=DEFAULT_MAXLEN, top_k=DEFAULT_TOPK)
                labels_pred.append(items[0][0])

            dist = pd.Series(labels_pred).value_counts().sort_values(ascending=False)
            st.markdown("#### توزيع الفئات المتوقعة")
            fig, ax = plt.subplots(figsize=(8,4))
            dist.plot(kind="bar", ax=ax, color="#38bdf8")
            ax.set_xlabel("الفئة"); ax.set_ylabel("العدد"); ax.set_title("توزيع التنبؤات")
            st.pyplot(fig, clear_figure=True)

            st.markdown("#### جدول")
            st.dataframe(dist.reset_index().rename(columns={"index":"الفئة", 0:"العدد"}), use_container_width=True)

# ==================== تذييل ====================
st.markdown(
    f"""
<div class="small-muted" style="margin-top:24px; text-align:center;">
  يستخدم المجلد <b>{DEFAULT_MODEL_DIR}</b> كمصدر للنموذج (يمكن تغييره من الشريط الجانبي).<br/>
  الرفع يدعم <b>PDF</b> و<b>TXT</b> و<b>DOCX</b> نصًا صريحًا كما طُلِب. الوضع الدفعي يتوقع عمود <code>text</code>.
</div>
""",
    unsafe_allow_html=True
)
