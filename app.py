# -----------------------------
# Phishing model (unchanged)
# -----------------------------
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

REPO_ID = "lwiz/louai-phishing-distilbert-uncased-finetuned"  # HF model repo id

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(REPO_ID)  # public HF: no token needed
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

# -----------------------------
# Gmail OAuth (hardened)
# -----------------------------
import urllib.parse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# >>> Values come from Streamlit Secrets (do NOT hardcode here) <<<
CLIENT_ID     = st.secrets["GOOGLE_CLIENT_ID"].strip()
CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"].strip()
REDIRECT_URI  = st.secrets["GOOGLE_REDIRECT_URI"].strip()   # e.g., "https://phidetector.streamlit.app/"
SCOPES        = st.secrets["SCOPES"].split()                # e.g., "https://www.googleapis.com/auth/gmail.readonly"

def build_flow():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
    )
    flow.redirect_uri = REDIRECT_URI
    return flow

st.subheader("Connect Gmail (read-only)")

# 1) Reuse credentials if present in this session
creds = None
if "creds_json" in st.session_state:
    creds = Credentials.from_authorized_user_info(st.session_state["creds_json"])
    st.session_state["token_exchanged"] = True  # never redeem the code again in this session

# 2) Handle the Google callback exactly once
code  = st.query_params.get("code")
state = st.query_params.get("state")

if code and not st.session_state.get("token_exchanged"):
    # Build the exact authorization_response Google called with (?code=...&state=...)
    auth_resp = REDIRECT_URI + "?" + urllib.parse.urlencode({k: v for k, v in st.query_params.items()})

    try:
        flow = build_flow()

        # If we initiated login earlier, validate state to avoid mismatches
        expected_state = st.session_state.get("oauth_state")
        if expected_state and expected_state != state:
            st.warning("OAuth state mismatch, restarting sign-in…")
            st.query_params.clear()  # drop the bad params
        else:
            # Redeem the one-time code using the full authorization response
            flow.fetch_token(authorization_response=auth_resp)
            creds = flow.credentials

            # Cache minimal creds for this session so reruns won't reuse the code
            st.session_state["creds_json"] = {
                "token": creds.token,
                "refresh_token": getattr(creds, "refresh_token", None),
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "scopes": SCOPES,
            }
            st.session_state["token_exchanged"] = True

            # Clear query params immediately so a rerun cannot attempt a second exchange
            st.query_params.clear()

    except Exception as e:
        st.error(f"OAuth error during token exchange: {e}")
        st.stop()

# 3) If we still don't have creds, start the flow and remember state
if not creds:
    flow = build_flow()
    auth_url, oauth_state = flow.authorization_url(
        access_type="offline",             # allows refresh_token on first consent
        include_granted_scopes=True,
        prompt="consent",
    )
    st.session_state["oauth_state"] = oauth_state
    st.link_button("Sign in with Google", auth_url)
    st.stop()

# 4) We have creds – build Gmail service and show messages
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
            userId="me", id=m["id"], format="metadata", metadataHeaders=["Subject", "From"]
        ).execute()
        headers = {h["name"]: h["value"] for h in full["payload"].get("headers", [])}
        subject = headers.get("Subject", "(no subject)")
        sender  = headers.get("From", "(unknown)")
        snippet = full.get("snippet", "")
        st.write(f"**{subject}**")
        st.write(f"From: {sender}")
        st.write(snippet[:200] + ("…" if len(snippet) > 200 else ""))
        st.divider()
