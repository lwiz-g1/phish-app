# -----------------------------
# Phishing model (unchanged)
# -----------------------------
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

REPO_ID = "lwiz/louai-phishing-distilbert-uncased-finetuned"

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(REPO_ID)
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
# Gmail OAuth (diagnostic)
# -----------------------------
import urllib.parse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def _get_secret(key: str) -> str:
    v = st.secrets[key]
    return str(v).strip().strip('"').strip("'")

CLIENT_ID     = _get_secret("GOOGLE_CLIENT_ID")
CLIENT_SECRET = _get_secret("GOOGLE_CLIENT_SECRET")
REDIRECT_URI  = _get_secret("GOOGLE_REDIRECT_URI")  # e.g., https://phidetector.streamlit.app/

_scopes_raw = st.secrets["SCOPES"]
if isinstance(_scopes_raw, str):
    parts = [p.strip() for chunk in _scopes_raw.split(",") for p in chunk.split()]
    SCOPES = [p for p in parts if p]
else:
    SCOPES = list(_scopes_raw)

def build_flow() -> Flow:
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

# Show what we will send (for debugging)
with st.expander("🔎 OAuth debug (what the app will send)"):
    st.write("Redirect URI used by the app:")
    st.code(REDIRECT_URI)
    st.write("Scopes:")
    st.code(SCOPES)

# 1) Reuse creds if already in session
creds = None
if "creds_json" in st.session_state:
    creds = Credentials.from_authorized_user_info(st.session_state["creds_json"])
    st.session_state["token_exchanged"] = True

# 2) Handle callback once
code  = st.query_params.get("code")
state = st.query_params.get("state")

if code and not st.session_state.get("token_exchanged"):
    auth_resp = REDIRECT_URI + "?" + urllib.parse.urlencode({k: v for k, v in st.query_params.items()})
    try:
        flow = build_flow()
        expected_state = st.session_state.get("oauth_state")
        if expected_state and expected_state != state:
            st.warning("OAuth state mismatch, restarting sign-in…")
            st.query_params.clear()
        else:
            flow.fetch_token(authorization_response=auth_resp)
            creds = flow.credentials
            st.session_state["creds_json"] = {
                "token": creds.token,
                "refresh_token": getattr(creds, "refresh_token", None),
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "scopes": SCOPES,
            }
            st.session_state["token_exchanged"] = True
            st.query_params.clear()
    except Exception as e:
        st.error(f"OAuth error during token exchange: {e}")
        st.stop()

# 3) No creds yet → start flow (booleans, not strings)
if not creds:
    flow = build_flow()
    auth_url, oauth_state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",   # boolean
        prompt="consent",
    )
    st.session_state["oauth_state"] = oauth_state

    # Show the actual URL we send to Google
    with st.expander("🔗 Authorization URL (exact)"):
        st.code(auth_url)

    st.link_button("Sign in with Google", auth_url)
    st.stop()

# 4) Call Gmail
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
