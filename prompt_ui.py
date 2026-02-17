import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from prompt_generator import get_prompt_template, save_prompt_to_json

# ------------------ Load ENV ------------------
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ------------------ LLM SETUP ------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=api_token,
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Research Paper Explainer", layout="centered")

st.header("📄 Research Paper Explainer")

paper_input = st.selectbox(
    "Select Research Paper",
    (
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    )
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (Detailed Explanation)"]
)

# ------------------ PROMPT TEMPLATE ------------------
prompt_template = get_prompt_template()

# ------------------ ACTION ------------------
if st.button("Summarize"):
    with st.spinner("Generating explanation..."):
        # Build prompt
        prompt = prompt_template.format(
            paper=paper_input,
            style=style_input,
            length=length_input
        )

        # Save prompt
        saved_prompt = save_prompt_to_json(
            paper_input,
            style_input,
            length_input,
            prompt
        )

        # LLM call
        response = model.invoke(prompt)

        st.subheader("📘 Explanation")
        st.write(response.content)

        st.subheader("🧾 Saved Prompt (JSON)")
        st.json(saved_prompt)
