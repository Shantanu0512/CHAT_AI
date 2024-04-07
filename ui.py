import streamlit as st
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

# Load the LLM model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate response
def generate_response(image, question):
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    return processor.token2json(sequence)

# Streamlit UI
st.title("Document Question Answering")
st.write("Upload an image and ask a question about it.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    question = st.text_input("Enter your question")

    if st.button("Generate Response"):
        if uploaded_image is not None and question:
            # Open the uploaded image
            image = Image.open(uploaded_image)

            # Generate response
            response = generate_response(image, question)

            st.success(response)
        else:
            st.warning("Please upload an image and enter a question.")
