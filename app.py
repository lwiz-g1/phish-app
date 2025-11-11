import json, torch, streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

REPO_ID = "lwiz/louai-phishing-distilbert-uncased-finetuned"  # <-- HF model repo id

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

import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

CLIENT_ID     = st.secrets["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI  = st.secrets["GOOGLE_REDIRECT_URI"]
SCOPES        = st.secrets["SCOPES"].split()

def _flow():
    return Flow.from_client_config(
        {"web":{
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri":"https://accounts.google.com/o/oauth2/auth",
            "token_uri":"https://oauth2.googleapis.com/token",
            "redirect_uris":[REDIRECT_URI],
        }},
        scopes=SCOPES
    )

st.subheader("Connect Gmail (read-only)")

code = st.query_params.get("code", None)
creds = None
if code:
    flow = _flow()
    flow.redirect_uri = REDIRECT_URI
    flow.fetch_token(code=code)
    creds = flow.credentials

if not creds or not creds.valid:
    flow = _flow()
    flow.redirect_uri = REDIRECT_URI
    auth_url, _ = flow.authorization_url(
        access_type="online", include_granted_scopes="true", prompt="consent"
    )
    st.link_button("Sign in with Google", auth_url)
    st.stop()

service = build("gmail", "v1", credentials=creds)
st.success("Signed in to Gmail ✅")

resp = service.users().messages().list(userId="me", q="newer_than:7d", maxResults=10).execute()
msgs = resp.get("messages", [])
if not msgs:
    st.info("No messages from the last 7 days.")
else:
    st.caption("Recent messages (subject + snippet)")
    for m in msgs:
        full = service.users().messages().get(
            userId="me", id=m["id"], format="metadata", metadataHeaders=["Subject","From"]
        ).execute()
        headers = {h["name"]:h["value"] for h in full["payload"].get("headers", [])}
        subject = headers.get("Subject","(no subject)")
        sender = headers.get("From","(unknown)")
        snippet = full.get("snippet","")
        st.write(f"**{subject}**")
        st.write(f"From: {sender}")
        st.write(snippet[:200] + ("…" if len(snippet)>200 else ""))
        st.divider()
