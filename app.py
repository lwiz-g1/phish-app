# ==============================
# Phishing Email Detector (Streamlit)
# - DistilBERT classifier
# - Gmail OAuth (read-only)
# - Threshold persistence + FP/FN logger
# - Header-based trust tweak to reduce FPs
# ==============================

# ---------- Imports ----------
import json
import re
import base64
import html
import io
import csv
import requests
import urllib.parse
from urllib.parse import urlparse

import torch
import streamlit as st
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


# ---------- Model ----------
REPO_ID = "lwiz/louai-phishing-distilbert-uncased-finetuned"


# ====== (B) Helper functions for better email text extraction & structured model input ======

URL_RE = re.compile(r"(https?://[^\s<>\"]+|www\.[^\s<>\"]+)", re.IGNORECASE)

def _b64url_decode(data: str) -> str:
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_all_text_parts(payload: dict) -> tuple[str, str]:
    """
    Returns (plain_text, html_text) aggregated from ALL nested MIME parts.
    """
    plain_chunks, html_chunks = [], []

    def walk(part: dict):
        if not isinstance(part, dict):
            return

        mime = (part.get("mimeType") or "").lower()
        body = part.get("body") or {}
        data = body.get("data")

        if data:
            decoded = _b64url_decode(data)
            if "text/plain" in mime:
                plain_chunks.append(decoded)
            elif "text/html" in mime:
                html_chunks.append(decoded)

        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload or {})
    return "\n".join(plain_chunks).strip(), "\n".join(html_chunks).strip()

def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = URL_RE.findall(text)

    norm = []
    for u in urls:
        if u.lower().startswith("www."):
            norm.append("http://" + u)
        else:
            norm.append(u)

    seen, out = set(), []
    for u in norm:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def extract_urls_from_html(html_text: str) -> list[str]:
    if not html_text:
        return []
    soup = BeautifulSoup(html_text, "lxml")

    urls = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if href:
            urls.append(href)

    urls.extend(extract_urls_from_text(html_text))

    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def clean_html_to_visible_text(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

def get_header(headers: list[dict], name: str) -> str:
    if not headers:
        return ""
    name_l = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == name_l:
            return (h.get("value") or "").strip()
    return ""

def build_model_input(subject: str, from_: str, reply_to: str, body_text: str, urls: list[str]) -> str:
    """
    Produces a consistent structured input string for the classifier.
    """
    urls_line = " | ".join((urls or [])[:15])
    parts = [
        f"[SUBJECT] {subject.strip() if subject else ''}",
        f"[FROM] {from_.strip() if from_ else ''}",
        f"[REPLY-TO] {reply_to.strip() if reply_to else ''}",
        f"[URLS] {urls_line}",
        f"[BODY] {body_text.strip() if body_text else ''}",
    ]
    return "\n".join(parts).strip()


@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    mdl.eval()
    thr_file = hf_hub_download(REPO_ID, filename="threshold.json")
    thr = json.load(open(thr_file))["threshold"]
    return tok, mdl, float(thr)

st.title("Phishing Email Detector")

tok, mdl, thr_saved = load_model()

# ----- Sticky UI threshold (defaults to 0.80 for fewer false alarms) -----
default_thr = st.session_state.get("ui_threshold", 0.80)
thr_ui = st.sidebar.slider(
    "Decision threshold",
    0.10, 0.95, float(default_thr), 0.01,
    help="Higher threshold = fewer false positives (but more false negatives)",
)
st.session_state["ui_threshold"] = thr_ui

# Quick note when user runs very high/low values
if thr_ui >= 0.85:
    st.sidebar.info("High precision mode")
elif thr_ui <= 0.40:
    st.sidebar.warning("High recall mode")

# ----- Paste-box demo -----
txt = st.text_area("Paste email subject + body:", height=220, placeholder="Subject line\n\nBody text…")
if st.button("Classify"):
    model_txt = build_model_input(
        subject="(manual input)",
        from_="",
        reply_to="",
        body_text=txt,
        urls=extract_urls_from_text(txt),
    )
    enc = tok(model_txt, truncation=True, padding=True, max_length=384, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
        prob = torch.softmax(out.logits, dim=1).numpy().ravel()[1].item()
    label = "PHISHING" if prob >= thr_ui else "LEGIT"
    st.metric("Prediction", label)
    st.json({"model_prob": prob, "prob_after_header_rules": prob, "threshold_used": thr_ui})
    st.progress(min(1.0, prob))


# ---------- Gmail OAuth ----------
def _get_secret(key: str) -> str:
    # IMPORTANT: edit in Streamlit Secrets, not here
    v = st.secrets[key]
    return str(v).strip().strip('"').strip("'")

CLIENT_ID     = _get_secret("GOOGLE_CLIENT_ID")      # secrets
CLIENT_SECRET = _get_secret("GOOGLE_CLIENT_SECRET")  # secrets
REDIRECT_URI  = _get_secret("GOOGLE_REDIRECT_URI")   # e.g., https://phidetector.streamlit.app/
_scopes_raw   = st.secrets["SCOPES"]                 # "https://www.googleapis.com/auth/gmail.readonly"

# Normalize SCOPES to a list
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

# Reuse creds if session has them
creds = None
if "creds_json" in st.session_state:
    creds = Credentials.from_authorized_user_info(st.session_state["creds_json"])
    st.session_state["token_exchanged"] = True

# Handle callback once
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

# Start flow (same-tab link)
if not creds:
    flow = build_flow()
    auth_url, oauth_state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = oauth_state

    st.markdown(
        f'<a href="{auth_url}" style="display:inline-block;padding:0.6rem 1rem;'
        'background:#0e5ef7;color:white;border-radius:0.5rem;text-decoration:none;">'
        'Sign in with Google</a>',
        unsafe_allow_html=True,
    )
    st.stop()


# ---------- Gmail helpers (kept for UI display headers fallback) ----------
def _decode_b64url(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode(errors="ignore")
    except Exception:
        return ""

def extract_text_from_payload(payload: dict) -> str:
    """Prefer text/plain; fallback to stripped HTML; walk MIME tree."""
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
    text = "\n\n".join([t.strip() for t in texts_plain if t.strip()])
    if not text:
        text = "\n\n".join([t.strip() for t in texts_html if t.strip()])
    return " ".join(text.split())

def header_get(payload: dict, name: str, default=""):
    for h in payload.get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", default)
    return default


def extract_email_features_from_gmail_message(full_msg: dict) -> dict:
    """
    Extract subject/from/reply-to + full text + urls using the (B) helpers.
    """
    payload = full_msg.get("payload") or {}
    headers_list = payload.get("headers") or []

    subject = get_header(headers_list, "Subject") or "(no subject)"
    from_ = get_header(headers_list, "From") or "(unknown)"
    reply_to = get_header(headers_list, "Reply-To") or ""

    plain_text, html_text = extract_all_text_parts(payload)

    if plain_text:
        body = " ".join(plain_text.split())
        urls = extract_urls_from_text(plain_text)
    else:
        body = clean_html_to_visible_text(html_text)
        urls = extract_urls_from_html(html_text)

    body = body[:6000]

    return {
        "subject": subject,
        "from": from_,
        "reply_to": reply_to,
        "body": body,
        "urls": urls,
    }


# ---------- Small FP reduction: trusted-sender header boost ----------
TRUSTED_DOMAINS = {
    "github.com", "google.com", "paypal.com", "microsoft.com",
    "apple.com", "googlemail.com", "amazon.com"
}

def header_trust_boost(headers: dict) -> float:
    """If SPF/DKIM pass AND From is a trusted domain, subtract 0.20 from prob."""
    auth = (headers.get("Authentication-Results", "") + " " + headers.get("Received-SPF", "")).lower()
    from_addr = headers.get("From", "").lower()
    has_pass = ("spf=pass" in auth) or ("dkim=pass" in auth)
    trusted  = any(d in from_addr for d in TRUSTED_DOMAINS)
    return -0.20 if (has_pass and trusted) else 0.0


# ---------- FP/FN logger ----------
if "label_log" not in st.session_state:
    st.session_state["label_log"] = []  # list of dicts

def log_example(example_id: str, text: str, model_prob: float, prob_adj: float, used_thr: float, predicted: str, true_label: str):
    st.session_state["label_log"].append({
        "id": example_id,
        "text": text[:4000],           # keep CSV manageable
        "model_prob": round(model_prob, 4),
        "prob_after_rules": round(prob_adj, 4),
        "threshold_used": round(used_thr, 2),
        "predicted": predicted,
        "true_label": true_label,      # "FP" (should be LEGIT) or "FN" (should be PHISHING)
    })

def download_log_button():
    if not st.session_state["label_log"]:
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(st.session_state["label_log"][0].keys()))
    writer.writeheader()
    writer.writerows(st.session_state["label_log"])
    st.download_button("Download FP/FN log (CSV)", buf.getvalue(), "label_log.csv", "text/csv")


# ---------- Signed in → list & classify ----------
service = build("gmail", "v1", credentials=creds)
st.success("Signed in to Gmail ✅")

# Sign out / revoke
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
    for k in ("creds_json", "token_exchanged", "oauth_state"):
        st.session_state.pop(k, None)
    st.query_params.clear()
    st.success("Signed out.")
    st.stop()

st.sidebar.button("Sign out / Clear session", on_click=sign_out)
download_log_button()

# Fetch last 10 messages
resp = service.users().messages().list(userId="me", q="newer_than:7d", maxResults=10).execute()
msgs = resp.get("messages", [])
if not msgs:
    st.info("No messages from the last 7 days.")
else:
    st.caption("Recent messages (subject + extracted body)")
    for m in msgs:
        full = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        payload = full.get("payload", {}) or {}
        headers_list = payload.get("headers", []) or []
        headers = {h.get("name",""): h.get("value","") for h in headers_list}

        features = extract_email_features_from_gmail_message(full)

        subject = features["subject"]
        sender = features["from"]
        body_text = features["body"]
        urls = features["urls"]

        preview = (body_text[:800] + ("…" if len(body_text) > 800 else "")) if body_text else full.get("snippet","")

        display_text = build_model_input(
            subject=subject,
            from_=sender,
            reply_to=features["reply_to"],
            body_text=body_text,
            urls=urls,
        )

        st.write(f"**{subject}**")
        st.write(f"From: {sender}")
        st.write(preview if preview else "(no content)")

        # ---- Classify with optional header-based trust tweak ----
        if st.button(f"Classify this #{m['id']}", key=m["id"]):
            enc = tok(display_text, truncation=True, padding=True, max_length=384, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**enc)
                prob = torch.softmax(out.logits, dim=1).numpy().ravel()[1].item()

            prob_adj = min(max(prob + header_trust_boost(headers), 0.0), 1.0)
            label = "PHISHING" if prob_adj >= thr_ui else "LEGIT"

            st.info(f"{label}  (model_prob={prob:.3f}, prob_after_rules={prob_adj:.3f}, threshold_used={thr_ui:.2f})")
            st.json({"model_prob": prob, "prob_after_header_rules": prob_adj, "threshold": thr_ui})

            # Quick feedback buttons to build your re-train set
            cols = st.columns(3)
            with cols[0]:
                if st.button("Mark as FALSE POSITIVE (should be LEGIT)", key=m["id"]+"_fp"):
                    log_example(m["id"], display_text, prob, prob_adj, thr_ui, label, "FP")
                    st.success("Logged as FP")
            with cols[1]:
                if st.button("Mark as FALSE NEGATIVE (should be PHISHING)", key=m["id"]+"_fn"):
                    log_example(m["id"], display_text, prob, prob_adj, thr_ui, label, "FN")
                    st.success("Logged as FN")
            with cols[2]:
                download_log_button()

        st.divider()
