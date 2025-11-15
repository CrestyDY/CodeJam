import os
import asyncio
import threading
from concurrent.futures import Future
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.ai.prompts import prompt1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "etc", ".env"))
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")

# Global state for persistent event loop and connection
_loop = None
_loop_thread = None
_client = None
_connection = None

def _start_background_loop():
    """Start a background event loop in a daemon thread"""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

def get_event_loop():
    """Get or create the persistent event loop"""
    global _loop, _loop_thread
    if _loop is None or not _loop.is_running():
        _loop_thread = threading.Thread(target=_start_background_loop, daemon=True)
        _loop_thread.start()
        # Wait a bit for loop to start
        import time
        time.sleep(0.1)
    return _loop

async def _get_persistent_connection():
    """Get or create a persistent connection to the Realtime API"""
    global _client, _connection
    
    if _client is None:
        _client = AsyncOpenAI(api_key=API_KEY)
    
    if _connection is None:
        try:
            _connection = await _client.realtime.connect(model="gpt-realtime").__aenter__()
            # Match official docs exactly - minimal session config
            await _connection.session.update(
                session={"type": "realtime", "output_modalities": ["text"]}
            )
        except Exception as e:
            print(f"Error creating connection: {e}")
            _connection = None
            raise
    
    return _connection

async def _get_llm_response_async(prompt_text: str):
    """Internal async function to get LLM response"""
    try:
        connection = await _get_persistent_connection()
        
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
            # Match official docs event handling
            if event.type == "response.output_text.delta":
                full_text += event.delta
                print(event.delta, flush=True, end="")
            
            elif event.type == "response.output_text.done":
                print()
            
            elif event.type == "response.done":
                break

        return full_text
    
    except Exception as e:
        # If connection fails, reset it and retry once
        global _connection
        _connection = None
        print(f"Connection error, retrying: {e}")
        
        # Retry with new connection
        connection = await _get_persistent_connection()
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
                full_text += event.delta
                print(event.delta, flush=True, end="")
            
            elif event.type == "response.output_text.done":
                print()
            
            elif event.type == "response.done":
                break
        
        return full_text

def get_llm_response(prompt_text: str):
    """Synchronous wrapper that submits to the persistent event loop"""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(_get_llm_response_async(prompt_text), loop)
    try:
        return future.result(timeout=30)  # 30 second timeout
    except Exception as e:
        print(f"Error in get_llm_response: {e}")
        raise

def create_prompt(sentence: str) -> str:
    """Generate prompt from user input using prompt1"""
    return prompt1(user_input=sentence)

def get_response(user_input):
    """Get response from LLM (now synchronous but uses persistent event loop)"""
    prompt_text = create_prompt(user_input)
    response = get_llm_response(prompt_text)
    return response

