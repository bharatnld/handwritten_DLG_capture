import streamlit as st
import json
import tempfile
from single_image import extract_text_and_schema_from_image

st.set_page_config(page_title="Shipment Document Parser", layout="wide")

st.title("📄 Shipment Document Parser (Gemini + OCR)")
st.markdown(
    "Upload a scanned image (JPEG or PNG) of a **CMR**, **delivery note**, or any shipping document. "
    "Both **printed and handwritten** elements will be parsed into structured JSON."
)

# File upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_image_path = tmp_file.name

    with st.spinner("🔍 Processing document... This may take a few seconds."):
        raw_text, structured = extract_text_and_schema_from_image(temp_image_path)


    st.subheader("📦 Structured JSON Output")
    st.json(structured)

    # Downloadable JSON
    json_str = json.dumps(structured, indent=2)
    st.download_button(
        label="⬇️ Download JSON",
        data=json_str,
        file_name="structured_document.json",
        mime="application/json"
    )
