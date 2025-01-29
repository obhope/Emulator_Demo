from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Qwen Tokenization API!"}

# Define request and response models
class TokenizationRequest(BaseModel):
    text: str
    max_length: int = 128  # Optional max length parameter

class TokenizationResponse(BaseModel):
    token_ids: List[int]
    decoded_tokens: List[str]
    token_count: int  # ✅ Ensures token_count is included in the response

# Tokenization function
def tokenize_text(text: str, max_length: int = 128) -> dict:
    """
    Tokenizes the input text using the DeepSeek-R1-Distill-Qwen-1.5B tokenizer.

    Args:
        text (str): The input text to tokenize.
        max_length (int): Maximum length of tokens.

    Returns:
        dict: A dictionary containing token IDs, decoded tokens, and token count.
    """
    tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    token_ids = tokens.input_ids.tolist()[0]  # Convert tensor to list
    decoded_tokens = [tokenizer.decode([token]) for token in token_ids]
    token_count = len(decoded_tokens) - 1  # ✅ Fix token count calculation

    return {
        "token_ids": token_ids,
        "decoded_tokens": decoded_tokens[1:],
        "token_count": token_count  # ✅ Ensure this is included
    }

# Tokenize text endpoint
@app.post("/tokenize", response_model=TokenizationResponse)
def tokenize(request: TokenizationRequest):
    """
    Endpoint to tokenize input text.

    Args:
        request (TokenizationRequest): Request body containing the text and max length.

    Returns:
        TokenizationResponse: Tokenized output including token count.
    """
    result = tokenize_text(request.text, request.max_length)
    return TokenizationResponse(**result)  # ✅ Returns token_count properly