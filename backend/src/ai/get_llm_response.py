import os
import asyncio
import threading
from concurrent.futures import Future
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.ai.prompts import prompt1, check_sentence_complete, casual_prompt, professional_prompt
from pathlib import Path

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
    return prompt1(user_input=sentence, tone=casual_prompt)

def get_response(user_input):
    """Get response from LLM (now synchronous but uses persistent event loop)"""
    prompt_text = create_prompt(user_input)
    response = get_llm_response(prompt_text)
    return response

def check_if_sentence_complete(user_input):
    """Check if the current input looks like a complete sentence using LLM"""
    prompt_text = check_sentence_complete(user_input, tone=casual_prompt)
    response = get_llm_response(prompt_text)
    return response

async def _speak_text_async(text: str):
    """Async function to speak text using OpenAI TTS"""
    global _client

    if _client is None:
        _client = AsyncOpenAI(api_key=API_KEY)

    try:
        speech_file_path = Path(__file__).parent / "speech.mp3"

        async with _client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=text,
        ) as response:
            await response.stream_to_file(speech_file_path)

        # Play the audio file using system command
        import platform
        import subprocess

        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.Popen(["afplay", str(speech_file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":
            # Try different audio players
            for player in ["mpg123", "ffplay", "aplay"]:
                try:
                    subprocess.Popen([player, str(speech_file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            subprocess.Popen(["start", str(speech_file_path)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"🔊 Speaking: {text}")

    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def speak_text(text: str):
    """Synchronous wrapper to speak text using OpenAI TTS"""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(_speak_text_async(text), loop)
    try:
        future.result(timeout=10)  # 10 second timeout for TTS
    except Exception as e:
        print(f"Error in speak_text: {e}")
