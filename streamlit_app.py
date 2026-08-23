"""Simple, voice-friendly Streamlit demo UI."""
import requests
import streamlit as st

st.set_page_config(page_title="SachCheck Demo", page_icon="🔎")
st.title("🔎 SachCheck: Verify a message or media")
st.caption("SachCheck assists verification ;  it does not establish truth.")
api = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")
kind = st.radio("What do you want to check?", ["Text", "Audio", "Video"], horizontal=True)
lang = st.selectbox("Result language", [("Hindi", "hi"), ("English", "en"), ("Tamil", "ta"), ("Telugu", "te"), ("Bengali", "bn")], format_func=lambda x:x[0])[1]
text = st.text_area("Paste or type the message") if kind == "Text" else ""
upload = None if kind == "Text" else st.file_uploader("Upload media", type=["wav","mp3","m4a"] if kind == "Audio" else ["mp4","avi","mov"])

if st.button("Check now", type="primary"):
    with st.spinner("Analysing… first use downloads models."):
        data = {"content_type": kind.lower(), "text": text, "output_lang": lang}
        files = {"file": (upload.name, upload.getvalue(), upload.type)} if upload else None
        response = requests.post(f"{api}/pipeline", data=data, files=files, timeout=300)
    if not response.ok: st.error(response.text)
    else:
        result = response.json(); st.subheader(f"Verdict: {result.get('verdict', result.get('label', 'unknown'))}")
        if result.get("transcript"): st.write("Transcript:", result["transcript"])
        if result.get("claims"):
            for claim in result["claims"]:
                st.warning(claim["sentence"]) if claim["label"] == "misleading" else st.write(claim["sentence"])
        if result.get("heatmap_url"): st.image(api + result["heatmap_url"], caption="Grad-CAM: areas influencing the model")
        explanation = result.get("translated_explanation") or result.get("warning") or "See the result above."
        st.info(explanation)
        audio = requests.post(f"{api}/tts", data={"text": explanation, "language": lang}).json()
        st.audio(api + audio["audio_url"])
