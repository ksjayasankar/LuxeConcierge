#core/llm_interface.py
import logging
import openai # Use the OpenAI library compatible with DeepSeek's API
import asyncio
from typing import List, Dict, Optional

# --- Absolute Imports ---
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, FALLBACK_ERROR_MESSAGE
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configure OpenAI Client ---
# Ensure API key and base URL are set correctly
if not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY is not set in environment variables!")
    # You might want to raise an error here or handle it depending on requirements
    # raise ValueError("Missing DeepSeek API Key")

# Initialize the client (works for OpenAI compatible APIs)
# Use AsyncOpenAI for async calls
client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
)
#logger.info(f"OpenAI client initialized for DeepSeek API at {DEEPSEEK_API_BASE}")


async def generate_text(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    stop: Optional[List[str]] = None,
    # Add other parameters like top_p if needed
) -> Optional[str]:
    """
    Calls the LLM API (DeepSeek via OpenAI library) to generate text based on messages.

    Args:
        messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
        model: The name of the LLM model to use.
        temperature: Controls randomness (0.0 to 2.0).
        max_tokens: Maximum number of tokens to generate.
        stop: Optional list of sequences to stop generation at.

    Returns:
        The generated text content as a string, or None if an error occurs.
    """
    logger.debug(f"Calling LLM model '{model}' with {len(messages)} messages.")
    # logger.debug(f"Messages: {messages}") # Uncomment for detailed debugging

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            # Add other parameters here if supported and needed
            # e.g., top_p=0.9
        )

        # Extract the response content
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            logger.debug(f"LLM Response received (Tokens: {response.usage.completion_tokens if response.usage else 'N/A'}): {content[:200]}...") # Log start of response
            return content.strip() if content else None
        else:
            logger.warning("LLM response did not contain any choices.")
            return None

    except openai.APIConnectionError as e:
        logger.error(f"API Connection Error: {e}")
    except openai.RateLimitError as e:
        logger.error(f"API Rate Limit Error: {e}")
        # Consider implementing exponential backoff here for retries
    except openai.AuthenticationError as e:
         logger.error(f"API Authentication Error: {e}. Check your API key.")
    except openai.APIStatusError as e:
        logger.error(f"API Status Error: status_code={e.status_code}, response={e.response}")
    except asyncio.TimeoutError:
         logger.error("API call timed out.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)

    return None # Return None on any error

# Example usage (can be tested standalone if needed)
# async def main_test():
#     test_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello, who are you?"}
#     ]
#     from luxury_leather_chatbot.config import CONVERSATIONAL_MODEL_NAME
#     response = await generate_text(test_messages, model=CONVERSATIONAL_MODEL_NAME)
#     if response:
#         print("LLM Test Response:", response)
#     else:
#         print("LLM Test Failed.")

# if __name__ == '__main__':
#     asyncio.run(main_test())