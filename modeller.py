

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import huggingface_hub
import transformers
import torch

def download_model(model_name, hf_token):
    """
    Downloads a model and its tokenizer from Hugging Face using the provided model name.

    Args:
        model_name (str): The name of the model to download from Hugging Face.

    Returns:
        model: The downloaded model.
        tokenizer: The corresponding tokenizer for the model.
    """

    # Login with HF Token
    # huggingface_hub.login(token=hf_token)

    # Download the model
    if torch.cuda.is_available():
        # model = AutoModel.from_pretrained(model_name,  torch_dtype=torch.float16, device_map="auto", use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch.float16, device_map="auto", use_auth_token=hf_token)
        model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        # tokenizer.use_default_system_prompt = False
        
        print(f"Model '{model_name}' and tokenizer downloaded successfully.")
        return model, tokenizer
    
    return "No GPU available on the PC to proceed... Please check if there's an error with GPU "


def initialize_pipeline():
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer
    )

if __name__ == "__main__":

    # Example usage
    #model_name = "bert-base-uncased"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    hf_token = "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"
    model, tokenizer = download_model(model_name, hf_token)
