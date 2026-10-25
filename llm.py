import json
import os
import re


MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def call_llm(system: str, user: str, expect_json=True, retries=2, temperature=0.2):
    """Wraps ollama.chat with schema-validation + self-correction retries.
    This single function is where 'malformed output' failures get absorbed
    so agents don't each have to reimplement retry logic."""
    import ollama  # lazy: the web dashboard must boot even without the SDK

    client = ollama.Client(host=OLLAMA_HOST)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = None
    for attempt in range(retries + 1):
        chat_kwargs = {
            "model": MODEL,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if expect_json:
            chat_kwargs["format"] = "json"

        try:
            resp = client.chat(**chat_kwargs)
        except Exception as ex:
            # If format="json" is not supported by the model or Ollama version, fallback without it
            if expect_json and "format" in chat_kwargs:
                chat_kwargs.pop("format", None)
                resp = client.chat(**chat_kwargs)
            else:
                raise ex

        # Handle both dict and object response types across ollama SDK versions
        if hasattr(resp, "message"):
            text = resp.message.content.strip()
        elif isinstance(resp, dict):
            text = resp["message"]["content"].strip()
        else:
            text = str(resp).strip()

        if not expect_json:
            return text

        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            # Feed the error back to the model and ask it to fix its own output
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    f"Your last response was not valid JSON ({last_error}). "
                    f"Reply with ONLY valid JSON matching the requested schema, nothing else."
                ),
            })
    raise ValueError(f"LLM failed to produce valid JSON after {retries} retries: {last_error}")


def _extract_json(text: str) -> dict:
    """Robustly parse JSON from text that may contain markdown fences or surrounding chatter."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response received from LLM")

    # 1. Direct parse attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    # 2. Extract from markdown code fences: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    # 3. Find outermost JSON object { ... }
    start_brace = text.find("{")
    end_brace = text.rfind("}")
    if start_brace != -1 and end_brace > start_brace:
        candidate = text[start_brace:end_brace + 1].strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    # 4. Find outermost JSON array [ ... ]
    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    if start_bracket != -1 and end_bracket > start_bracket:
        candidate = text[start_bracket:end_bracket + 1].strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    # Fallback to direct json.loads to trigger standard JSONDecodeError with position
    return json.loads(text)


def _strip_fences(text: str) -> str:
    """Legacy helper for backwards compatibility."""
    try:
        result = _extract_json(text)
        return json.dumps(result)
    except Exception:
        return text.strip()
