from openai import AzureOpenAI
from config.settings import get_settings

settings = get_settings()

client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version
)

response = client.chat.completions.create(
    model=settings.azure_deployment_chat,
    messages=[
        {
            "role": "system",
            "content": "You are a biomarker discovery assistant for Solid Biosciences."
        },
        {
            "role": "user",
            "content": "Say hello and confirm you are connected to the Mtx project."
        }
    ],
    max_tokens=100
)

print("✓ Connection successful")
print("─" * 40)
print(response.choices[0].message.content)
print("─" * 40)
print(f"Model used : {response.model}")
print(f"Tokens used: {response.usage.total_tokens}")