import validators
import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredImageLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import pytesseract

# Streamlit app
st.subheader('Vision Mate')

# Get OpenAI API key and input type
with st.sidebar:
    openai_api_key = st.text_input("Genini API key", value="", type="password")
    st.caption("*If you don't have an OpenAI API key, get it [here]().*")
    model = st.selectbox("OpenAI chat model", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"))
    st.caption("*If the article is long, choose gpt-3.5-turbo-16k.*")
    input_type = st.selectbox("Input type", ("URL", "Image"))

# If 'Summarize' button is clicked
if st.button("Describe"):
    # Validate inputs
    if not openai_api_key.strip():
        st.error("Please provide the OpenAI API key.")
    elif input_type == "URL":
        url = st.text_input("URL", label_visibility="collapsed")
        if not url.strip() or not validators.url(url):
            st.error("Please enter a valid URL.")
    elif input_type == "Image":
        image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if image is None:
            st.error("Please upload an image.")

    try:
        with st.spinner("Please wait..."):
            if input_type == "URL":
                # Load URL data
                loader = UnstructuredURLLoader(urls=[url])
            elif input_type == "Image":
                # Use Tesseract to extract text from the image
                image = Image.open(image)
                text_from_image = pytesseract.image_to_string(image)
                loader = UnstructuredImageLoader(texts=[text_from_image])

            data = loader.load()

            # Initialize the ChatOpenAI module, load and run the summarize chain
            llm = ChatOpenAI(temperature=0, model=model, openai_api_key=openai_api_key)
            prompt_template = """Write a summary of the following in 250-300 words:
                    
                {text}

            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(data)

            st.success(summary)
    except Exception as e:
        st.exception(f"Exception: {e}")
