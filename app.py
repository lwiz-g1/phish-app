import json, torch, streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

REPO_ID = "louai-phishing-distilbert-uncased-finetuned"  # <-- HF model repo id

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(REPO_ID)               # public HF: no token needed
    mdl = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    mdl.eval()
    thr_file = hf_hub_download(REPO_ID, filename="threshold.json")
    thr = json.load(open(thr_file))["threshold"]
    return tok, mdl, float(thr)

st.title("Phishing Email Detector (DistilBERT)")
tok, mdl, thr = load_model()
st.caption(f"Decision threshold: {thr:.2f}")

txt = st.text_area("Paste email subject + body:", height=220)
if st.button("Classify"):
    enc = tok(txt, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
        prob = torch.softmax(out.logits, dim=1).numpy().ravel()[1].item()
    label = "PHISHING" if prob >= thr else "LEGIT"
    st.metric("Prediction", label)
    st.json({"phishing_prob": prob, "threshold": thr})
    st.progress(min(1.0, prob))
