import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS

# -----------------------------
# TITLE
# -----------------------------

st.title("Universal Image Recognition AI")

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


processor, model = load_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded = st.file_uploader(
    "Upload Any Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PROCESS IMAGE
# -----------------------------

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Recognizing Image..."):

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        output = model.generate(
            **inputs,
            max_new_tokens=20,
            repetition_penalty=2.0,
            no_repeat_ngram_size=2
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

    # -----------------------------
    # TEXT OUTPUT
    # -----------------------------

    english_text = f"This is {caption}"

    

    st.success("Recognition Complete!")

    st.subheader("English:")
    st.write(english_text)



    # -----------------------------
    # ENGLISH VOICE
    # -----------------------------

    tts_en = gTTS(
        text=english_text,
        lang="en"
    )

    tts_en.save("english.mp3")

    audio_en = open("english.mp3", "rb")

    st.subheader("English Voice")

    st.audio(
        audio_en.read(),
        format="audio/mp3"
    )
