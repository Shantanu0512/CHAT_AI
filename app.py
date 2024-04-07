import streamlit as st
from PIL import Image
import numpy as np
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import torch
def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome to Maa Ki Chu!</h1>", unsafe_allow_html=True)
    
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)     
   
    image_file_buffer = st.camera_input("")
    
    if image_file_buffer is not None:
        image =   Image.open(image_file_buffer)
       
        # img_array = np.array(image)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for i  in range(len(st.session_state.messages)):
            agent = st.session_state.messages[i]["role"]
            agent_content = st.session_state.messages[i]["content"]
            
            with st.chat_message(agent):
                st.write(agent_content)
        
        user_inputs = st.chat_input("Please enter your question regarding the image here...")
        if user_inputs:
            with st.chat_message("user"):
                st.markdown(user_inputs)
                
            st.session_state.messages.append({"role": "user", "content": user_inputs})
            task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
            question = user_inputs
            prompt = task_prompt.replace("{user_input}", question)
            inputs = processor.tokenizer(user_inputs, add_special_tokens=False, return_tensors="pt").input_ids
            pixel_values = processor(image, return_tensors="pt").pixel_values
            outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=inputs.to(device),
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
            bot_response = processor.token2json(sequence)
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
    
    
if __name__ == "__main__":
    main()