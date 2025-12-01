# -----------------------------
# Phishing model (unchanged core, + threshold slider)
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

st.title("Phishing Email Detector")
tok, mdl, thr = load_model()
st.caption(f"Saved decision threshold: {thr:.2f}")

# NEW: safer, adjustable threshold for demo/testing
thr_ui = st.sidebar.slider(
    "Decision threshold",
    0.10, 0.90, float(thr), 0.01,
    help="Lower = catch more phishing (higher recall), Higher = fewer false alarms (higher precision)."
)

txt = st.text_area("Paste email subject + body:", height=220)
if st.button("Classify"):
    enc = tok(txt, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
        prob = torch.softmax(out.logits, dim=1).numpy().ravel()[1].item()
    label = "PHISHING" if prob >= thr_ui else "LEGIT"
    st.metric("Prediction", label)
    st.json({"phishing_prob": prob, "threshold_used": thr_ui})
    st.progress(min(1.0, prob))

# -----------------------------
# Gmail OAuth (hardened)
# -----------------------------
import urllib.parse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def _get_secret(key: str) -> str:
    v = st.secrets[key]
    # NEW: trim stray quotes/spaces from Secrets UI
    return str(v).strip().strip('"').strip("'")

CLIENT_ID     = _get_secret("GOOGLE_CLIENT_ID")     # EDIT IN STREAMLIT SECRETS, NOT HERE
CLIENT_SECRET = _get_secret("GOOGLE_CLIENT_SECRET") # EDIT IN STREAMLIT SECRETS, NOT HERE
REDIRECT_URI  = _get_secret("GOOGLE_REDIRECT_URI")  # e.g., https://phidetector.streamlit.app/
_scopes_raw   = st.secrets["SCOPES"]                # e.g., "https://www.googleapis.com/auth/gmail.readonly"

# Normalize scopes (string → list)
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

# 1) Reuse creds if already in session
creds = None
if "creds_json" in st.session_state:
    creds = Credentials.from_authorized_user_info(st.session_state["creds_json"])
    st.session_state["token_exchanged"] = True  # NEW: never try to redeem again in this session

# 2) Handle callback exactly once
code  = st.query_params.get("code")
state = st.query_params.get("state")

if code and not st.session_state.get("token_exchanged"):
    # NEW: build the exact authorization_response Google called with
    auth_resp = REDIRECT_URI + "?" + urllib.parse.urlencode({k: v for k, v in st.query_params.items()})
    try:
        flow = build_flow()
        expected_state = st.session_state.get("oauth_state")
        if expected_state and expected_state != state:
            st.warning("OAuth state mismatch, restarting sign-in…")
            st.query_params.clear()
        else:
            flow.fetch_token(authorization_response=auth_resp)  # redeem ONCE
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
            st.query_params.clear()  # NEW: prevent double redemption on rerun
    except Exception as e:
        st.error(f"OAuth error during token exchange: {e}")
        st.stop()

# 3) No creds yet → start flow (IMPORTANT: booleans, not strings)
if not creds:
    flow = build_flow()
    auth_url, oauth_state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes=True,   # NEW: boolean True (not "true")
        prompt="consent",
    )
    st.session_state["oauth_state"] = oauth_state

    # NEW: same-tab link (avoids opening a new window)
    st.markdown(
        f'<a href="{auth_url}" style="display:inline-block;padding:0.6rem 1rem;'
        'background:#0e5ef7;color:white;border-radius:0.5rem;text-decoration:none;">'
        'Sign in with Google</a>',
        unsafe_allow_html=True,
    )
    st.stop()

# -----------------------------
# Gmail helpers — complete body extraction
# -----------------------------
# NEW: robust HTML→text extraction to avoid "incomplete emails"
import base64, html
from bs4 import BeautifulSoup

def _decode_b64url(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode(errors="ignore")
    except Exception:
        return ""

def extract_text_from_payload(payload: dict) -> str:
    """Walk Gmail MIME tree. Prefer text/plain; fallback to stripped HTML."""
    texts_plain, texts_html = [], []

    def walk(p):
        if not isinstance(p, dict):
            return
        if "parts" in p:
            for sp in p.get("parts", []):
                walk(sp)
        else:
            mime = p.get("mimeType", "") or ""
            data = (p.get("body", {}) or {}).get("data")
            if not data:
                return
            raw = _decode_b64url(data)
            if "text/plain" in mime:
                texts_plain.append(html.unescape(raw))
            elif "text/html" in mime:
                texts_html.append(BeautifulSoup(raw, "html.parser").get_text(" ", strip=True))

    walk(payload)

    # prefer plain text
    text = "\n\n".join([t.strip() for t in texts_plain if t.strip()])
    if not text:
        text = "\n\n".join([t.strip() for t in texts_html if t.strip()])
    # normalize whitespace; keep reasonable length (model sees 256 tokens max)
    text = " ".join(text.split())
    return text

def header_get(payload: dict, name: str, default=""):
    for h in payload.get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", default)
    return default

# -----------------------------
# Signed in → list & classify messages
# -----------------------------
service = build("gmail", "v1", credentials=creds)
st.success("Signed in to Gmail ✅")

# NEW: sidebar sign-out / clear session
import requests
def sign_out():
    try:
        token = st.session_state.get("creds_json", {}).get("token")
        if token:
            requests.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": token},
                headers={"content-type": "application/x-www-form-urlencoded"},
                timeout=5,
            )
    except Exception:
        pass
    st.session_state.pop("creds_json", None)
    st.session_state.pop("token_exchanged", None)
    st.session_state.pop("oauth_state", None)
    st.query_params.clear()
    st.success("Signed out.")
    st.stop()

st.sidebar.button("Sign out / Clear session", on_click=sign_out)

# Fetch last 10 messages and show complete text (subject + extracted body)
resp = service.users().messages().list(userId="me", q="newer_than:7d", maxResults=10).execute()
msgs = resp.get("messages", [])
if not msgs:
    st.info("No messages from the last 7 days.")
else:
    st.caption("Recent messages (subject + extracted body)")
    for m in msgs:
        full = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        payload = full.get("payload", {}) or {}

        subject = header_get(payload, "Subject", "(no subject)")
        sender  = header_get(payload, "From", "(unknown)")
        body_text = extract_text_from_payload(payload)

        preview = (body_text[:800] + ("…" if len(body_text) > 800 else "")) if body_text else full.get("snippet","")
        display_text = f"{subject}\n\n{preview}".strip()

        st.write(f"**{subject}**")
        st.write(f"From: {sender}")
        st.write(preview if preview else "(no content)")

        # NEW: classify this message using the same model + adjustable threshold
        if st.button(f"Classify this #{m['id']}", key=m["id"]):
            enc = tok(display_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**enc)
                prob = torch.softmax(out.logits, dim=1).numpy().ravel()[1].item()
            label = "PHISHING" if prob >= thr_ui else "LEGIT"
            st.info(f"{label}  (p={prob:.3f}, threshold_used={thr_ui:.2f})")

        st.divider()
