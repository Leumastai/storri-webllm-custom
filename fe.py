import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as transformers_pipeline
import torch

# Function to download the model and tokenizer
def download_model(model_name, hf_token):
    """
    Downloads a model and its tokenizer from Hugging Face using the provided model name.
    Args:
        model_name (str): The name of the model to download from Hugging Face.
        hf_token (str): The Hugging Face authentication token.
    Returns:
        model: The downloaded model.
        tokenizer: The corresponding tokenizer for the model.
    """
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )
        model.cuda()
        return model, tokenizer
    else:
        st.error("No GPU available. Please check your GPU configuration.")
        return None, None

# Function to initialize the pipeline
def initialize_pipeline(model, tokenizer):
    """
    Initializes a text generation pipeline using the specified model and tokenizer.
    Args:
        model: The model to use for the pipeline.
        tokenizer: The tokenizer corresponding to the model.
    Returns:
        pipeline: The text generation pipeline.
    """
    return transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# Streamlit app
def main():
    st.title("LLM Text Generation")
    st.write("An app to generate text using a pre-trained Llama model from Hugging Face.")

    # Initialize session state variables
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

    # Hugging Face model configuration
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    hf_token = "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"  # Replace with your Hugging Face token

    # Download the model and tokenizer
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Downloading model and tokenizer..."):
            model, tokenizer = download_model(model_name, hf_token)
            if model and tokenizer:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("Model and tokenizer downloaded successfully.")

    # Initialize the pipeline
    if st.session_state.pipeline is None and st.session_state.model and st.session_state.tokenizer:
        with st.spinner("Initializing text generation pipeline..."):
            st.session_state.pipeline = initialize_pipeline(st.session_state.model, st.session_state.tokenizer)
            st.success("Pipeline initialized successfully.")

    # Text input for the user to provide input for text generation
    user_input = st.text_input("Enter a prompt for the LLM:", "Hi! Tell me about Llamas!")
    
    # Button to run the inference
    if st.button("Generate Text"):
        if st.session_state.pipeline is not None:
            with st.spinner("Generating text..."):
                # Run inference using the pipeline
                try:
                    sequences = st.session_state.pipeline(
                        user_input,
                        max_length=50,  # Adjust the length according to your needs
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1
                    )
                    generated_text = sequences[0]["generated_text"]
                    st.write(f"**Generated Text:** {generated_text}")
                except Exception as e:
                    st.error(f"Error during text generation: {str(e)}")
        else:
            st.error("Pipeline is not initialized.")

if __name__ == "__main__":
    main()
