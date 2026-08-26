# Using the model via API

OpenAI-compatible endpoint.

- **Base URL:** `https://resolution-andreas-alerts-blah.trycloudflare.com/v1`
- **API key:** `local-key`
- **Model:** `Qwen/Qwen3.6-35B-A3B`

```bash
pip install openai
```

## Text

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://resolution-andreas-alerts-blah.trycloudflare.com/v1",
    api_key="local-key",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "Explain KV caches in one paragraph."}],
    max_tokens=300,
)
print(resp.choices[0].message.content)
```

## Image (local file)

```python
import base64, mimetypes
from openai import OpenAI

path = "photo.jpg"
mime = mimetypes.guess_type(path)[0] or "image/jpeg"
b64 = base64.b64encode(open(path, "rb").read()).decode()
data_url = f"data:{mime};base64,{b64}"

client = OpenAI(
    base_url="https://resolution-andreas-alerts-blah.trycloudflare.com/v1",
    api_key="local-key",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Describe this image."},
        ],
    }],
    max_tokens=300,
)
print(resp.choices[0].message.content)
```
