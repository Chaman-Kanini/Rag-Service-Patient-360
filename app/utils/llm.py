import time as _time
from openai import AzureOpenAI
from app.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    CHATGPT_MODEL
)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)


def call_llm(prompt: str, timeout: int = 600, max_retries: int = 2, max_tokens: int = 64000) -> str:
    """Call Azure OpenAI LLM with a prompt. Retries on empty responses."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant AI that processes clinical documents and answers questions accurately. You MUST respond with valid JSON when asked for JSON output. Never return empty responses."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens,
                timeout=timeout
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            print(f"WARNING: LLM returned empty response (attempt {attempt + 1}/{max_retries + 1})")
            if response.choices[0].finish_reason:
                print(f"  finish_reason: {response.choices[0].finish_reason}")
            last_error = "LLM returned empty response"
            if attempt < max_retries:
                _time.sleep(2 * (attempt + 1))
        except Exception as e:
            last_error = str(e)
            print(f"WARNING: LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {last_error}")
            if attempt < max_retries:
                _time.sleep(2 * (attempt + 1))
    raise Exception(f"Error calling LLM after {max_retries + 1} attempts: {last_error}")
