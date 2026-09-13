from google import genai
from google.genai import errors
from config import GEMINI_API_KEYS


class GeminiHelper:
    """
    A helper class to interact with the Gemini API for generating insights.
    Automatically rotates through multiple API keys when one hits its rate limit.
    """

    def __init__(self, api_keys=None):
        self.api_keys = api_keys if api_keys else GEMINI_API_KEYS
        self.current_key_index = 0
        self.total_keys = len(self.api_keys) if self.api_keys else 0

        if self.total_keys == 0:
            raise ValueError("No API keys provided. Check your config.py")

        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

    def switch_api_key(self):
        """
        Switch to the next API key in the list. Loops back to the first key
        after the last one.
        """
        self.current_key_index = (self.current_key_index + 1) % self.total_keys
        new_api_key = self.api_keys[self.current_key_index]
        self.client = genai.Client(api_key=new_api_key)
        print(f"Switched to API key index: {self.current_key_index}")

    def get_insight(self, data_summary, prompt, model_name="gemini-3.1-flash-lite"):
        full_prompt = prompt.replace("<DATA_SUMMARY>", data_summary)

        attempts_left = self.total_keys

        while attempts_left > 0:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                )
                return response.text.strip()

            except errors.ClientError as e:
                if e.code == 429:
                    print(f"Key index {self.current_key_index} hit rate limit. Trying next key.")
                    self.switch_api_key()
                    attempts_left -= 1
                    continue
                else:
                    print(f"Gemini client error: {e}")
                    return None

            except Exception as e:
                print(f"Gemini call failed: {e}")
                return None

        print("All API keys are exhausted or rate-limited.")
        return None