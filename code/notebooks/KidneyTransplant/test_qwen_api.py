from openai import OpenAI

client = OpenAI(
    base_url="https://resolution-andreas-alerts-blah.trycloudflare.com/v1",
    api_key="local-key",
)

print("Client created successfully.")

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[
        {
            "role": "user",
            "content": (
                "Return a valid JSON object with a status field "
                "containing the value working."
            ),
        }
    ],
    max_tokens=500,
    temperature=0,
    )

print(response.choices[0].message.content)
