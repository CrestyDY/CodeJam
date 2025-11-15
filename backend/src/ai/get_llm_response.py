import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.prompts import prompt1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "etc", ".env"))
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")

async def get_llm_response_realtime(prompt_text: str):
    client = AsyncOpenAI(api_key=API_KEY)

    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(
            session={"type": "realtime", "output_modalities": ["text"]}
        )

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}],
            }
        )

        await connection.response.create()

        full_text = ""
        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
                full_text += event.delta
            elif event.type == "response.output_text.done":
                print()
            elif event.type == "response.done":
                break

        return full_text

def create_prompt(sentence: str) -> str:
    """Generate prompt from user input using prompt1"""
    return prompt1(user_input=sentence)

async def get_response(user_input):
    prompt_text = create_prompt(user_input)
    response = await get_llm_response_realtime(prompt_text)
    print("\nFinal response:", response)

if __name__ == "__main__":
    asyncio.run(get_response("STM, Strike, Cancel"))

