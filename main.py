# Standard library imports
import logging
import os
import re
import threading
import time
import warnings

# Third-party imports
import numpy as np
import soundcard as sc
import torch
import whisper
from soundcard import SoundcardRuntimeWarning
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI

# Initialize logging and warnings
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('app.log', encoding='utf-8')  # Specify UTF-8 encoding
    ]
)

# --------------Initialize Whisper Model----------------------
def load_whisper_model(model_size="tiny"):
    """
    Load Whisper model from local path or download if not available
    Args:
        model_size: Size of the model ('tiny', 'base', 'small', 'medium', 'large')
    Returns:
        Loaded Whisper model
    """
    # Validate model size
    valid_sizes = ['tiny', 'base', 'small', 'medium', 'large']
    if model_size not in valid_sizes:
        raise ValueError(f"Invalid model size. Must be one of: {valid_sizes}")

    # Set device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model path
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_size)
    os.makedirs(model_path, exist_ok=True)

    try:
        # Try loading from local cache
        model = whisper.load_model(model_size, download_root=model_path, device=device)
        logging.info(f"Model ***[{model_size}]*** loaded from local cache on {device.upper()}")
    except (FileNotFoundError, RuntimeError) as e:
        logging.warning(f"Cannot load model locally: {e}")
        logging.info("Downloading model...")
        try:
            model = whisper.load_model(model_size, device=device)
            logging.info(f"Model download completed on {device.upper()}")
            # Save model to cache
            model.save_pretrained(model_path)
            logging.info(f"Model cached locally at {model_path}")
        except Exception as download_error:
            logging.error(f"Failed to download model: {download_error}")
            raise
    
    return model

# --------------Audio Configuration----------------------
# Audio parameters
CLEAR_CONSOLE = False  # Switch to clear the console
RATE = 16000          # Sample rate
CHUNK = 4096          # Buffer size
SILENCE_THRESHOLD = 0.03  # Threshold for silence detection
SILENCE_CHUNKS = 3    # Number of consecutive silent chunks to trigger processing
MIN_AUDIO_LENGTH = 0.8  # Minimum valid audio length (seconds)
MAX_AUDIO_LENGTH = 15   # Maximum audio segment length (seconds)
MAX_LENGTH = 500      # Reduced maximum recorded text length

# Initialize global variables
audio_buffer = np.empty((0,), dtype=np.float32)
is_recording = True   # Global flag to control recording
final_transcription = ""  # Store final concatenated result
translator = None     # Global translator object
last_sentence = ""    # Store the last sentence
processing_buffer = None  # Buffer for processing
buffer_lock = threading.Lock()  # Lock for buffer synchronization
processing_thread = None  # Thread for audio processing
# Initialize translation model (lazy loading will handle it)
translator = None

# 添加翻译模型选择配置
TRANSLATION_MODEL = "local"  # 可选值: "kimi" 或 "local"
KIMI_API_KEY = "your-kimi-api-key"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"

# 初始化 Kimi 客户端
kimi_client = OpenAI(
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASE_URL,
)

def is_sentence_end(text):
    """Check if text ends with proper sentence termination punctuation"""
    if not text:
        return False
    return text.rstrip().endswith(('.', '?', '!', '。', '？', '！'))

def process_audio(audio_data):
    """Process audio data through Whisper model for transcription"""
    duration = len(audio_data) / RATE
    if duration < MIN_AUDIO_LENGTH:
        return ""
    try:
        # Convert multi-channel to mono and ensure float32 format
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        audio_data = audio_data.astype(np.float32)
        
        # Ensure audio data length is a multiple of 16000 (1 second)
        if len(audio_data) % 16000 != 0:
            padding_length = 16000 - (len(audio_data) % 16000)
            audio_data = np.pad(audio_data, (0, padding_length), mode='constant')
        
        transcription = transcription_model.transcribe(audio_data)
        return transcription['text']  # Extract text from the dictionary
    except Exception as e:
        #logging.error(f"Transcription failed: {e}")
        return ""

def smart_append_text(current_text, new_text):
    """
    Intelligently append new text to current text, avoiding duplicates and handling sentence breaks
    Args:
        current_text: Existing transcribed text
        new_text: New text to append
    Returns:
        Combined text with proper spacing and formatting
    """
    if not new_text:
        return current_text
    
    new_text = new_text.lstrip()
    if not current_text:
        return new_text
    
    # Check for overlapping text
    overlap_length = 0
    max_check = min(len(current_text), len(new_text))
    
    for i in range(1, max_check + 1):
        if current_text[-i:] == new_text[:i]:
            overlap_length = i
    
    if overlap_length > 0:
        return current_text + new_text[overlap_length:]
    
    if not current_text.endswith((' ', '.', '?', '!', ',', ';', ':', '-')):
        return current_text + " " + new_text
    
    return current_text + new_text

def translate_with_kimi(text):
    """使用 Kimi API 进行翻译"""
    if not text or text.isspace():
        return ""
    
    try:
        completion = kimi_client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "你是 Kimi，一个中英翻译官，可以将英文翻译成中文。回复只输出答案，不要输出多余的内容。"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Kimi translation failed: {e}")
        return text

def translate_to_chinese(text):
    """根据配置选择翻译模型"""
    global translator
    
    if not text or text.isspace():
        return ""
    
    if TRANSLATION_MODEL == "kimi":
        return translate_with_kimi(text)
    
    # 以下是原有的本地模型翻译逻辑
    if translator is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "translation_model/opus-mt-en-zh"
        )
        os.makedirs(model_path, exist_ok=True)
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            logging.info(f"Translation model loaded locally on {device.upper()}")
        except (OSError, ValueError) as e:
            logging.warning(f"Local model load failed: {e}")
            logging.info("Downloading translation model from mirror...")
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-zh", 
                    timeout=30
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-zh", 
                    timeout=30
                ).to(device)
            except Exception as download_error:
                logging.error(f"Model download failed: {download_error}")
                raise

            try:
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                logging.info(f"Translation model cached locally (device: {device})")
            except Exception as save_error:
                logging.error(f"Failed to save model: {save_error}")
        
        translator = (tokenizer, model)
    
    tokenizer, model = translator
    
    try:
        # Split long text into chunks to avoid memory issues
        max_length = 512
        text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        translated_chunks = []
        
        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=max_length, truncation=True).to(model.device)
            translated = model.generate(**inputs, max_length=max_length)
            translated_chunks.append(tokenizer.decode(translated[0], skip_special_tokens=True))
        
        return "".join(translated_chunks)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.warning("GPU memory exhausted, falling back to CPU")
            model.to("cpu")
            inputs = inputs.to("cpu")
            translated = model.generate(**inputs, max_length=max_length)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            logging.error(f"Translation failed: {e}")
            return text

def translate_to_chinese_async(text, callback):
    """Run translation in a separate thread"""
    def _translate():
        result = translate_to_chinese(text)
        callback(result)
    
    threading.Thread(target=_translate, daemon=True).start()

def detect_speech_activity(data, threshold=SILENCE_THRESHOLD):
    """Detect voice activity in audio buffer"""
    energy = np.mean(np.abs(data))
    return energy > threshold

def process_buffer_async(buffer_data):
    """Asynchronous audio processing pipeline"""
    global final_transcription
    
    text = process_audio(buffer_data)
    if text:
        with buffer_lock:
            final_transcription = smart_append_text(final_transcription, text)
            
            if len(final_transcription) > MAX_LENGTH:
                final_transcription = final_transcription[-(MAX_LENGTH // 3):]
        
        latest_text = final_transcription.strip()
        
        if latest_text and (not hasattr(process_buffer_async, 'last_translated') or 
                           latest_text != process_buffer_async.last_translated):
            def _print_translation(translated_text):
                if CLEAR_CONSOLE and latest_text:
                    os.system('cls' if os.name == 'nt' else 'clear')
                # Improved output formatting
                print("\n" + "=" * 50)
                print(f"English: {latest_text}")
                print("-" * 50)
                print(f"Chinese: {translated_text}")
                print("=" * 50 + "\n")
                process_buffer_async.last_printed = (latest_text, translated_text)
            
            translate_to_chinese_async(latest_text, _print_translation)
            process_buffer_async.last_translated = latest_text

def record_audio():
    """
    Main recording function that captures and processes audio in real-time
    """
    global audio_buffer, is_recording, processing_thread
    
    with loopback.recorder(samplerate=RATE) as mic:
        consecutive_silence_chunks = 0
        speech_detected = False
        last_processed_time = time.time()
        
        while is_recording:
            data = mic.record(numframes=CHUNK)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            has_speech = detect_speech_activity(data)
            
            if has_speech:
                speech_detected = True
                consecutive_silence_chunks = 0
                audio_buffer = np.concatenate((audio_buffer, data))
            else:
                if speech_detected:
                    consecutive_silence_chunks += 1
                    audio_buffer = np.concatenate((audio_buffer, data))
            
            current_time = time.time()
            buffer_duration = len(audio_buffer) / RATE
            
            should_process = (
                (speech_detected and consecutive_silence_chunks >= SILENCE_CHUNKS) or
                buffer_duration > MAX_AUDIO_LENGTH or
                (speech_detected and current_time - last_processed_time > 5)
            )
            
            if should_process and len(audio_buffer) > MIN_AUDIO_LENGTH * RATE:
                # 创建音频数据的副本用于处理
                process_data = np.copy(audio_buffer)
                
                # 启动新的处理线程
                if processing_thread is not None and processing_thread.is_alive():
                    processing_thread.join(timeout=0.1)
                
                processing_thread = threading.Thread(
                    target=process_buffer_async,
                    args=(process_data,),
                    daemon=True
                )
                processing_thread.start()
                
                # 重置录音缓冲区和状态
                audio_buffer = np.empty((0,), dtype=np.float32)
                speech_detected = False
                consecutive_silence_chunks = 0
                last_processed_time = current_time

# Main execution
if __name__ == "__main__":
    # Select model size
    model_size = "base"  # Can be "tiny", "base"，"small", "medium", "large"
    transcription_model = load_whisper_model(model_size)  # Use the selected model

    # Get audio devices
    speakers = sc.all_speakers()
    microphones = sc.all_microphones()

    # Log available devices
    logging.info("\nAvailable speakers:")
    for speaker in speakers:
        logging.info(f"- {speaker.name}")

    logging.info("\nAvailable microphones:")
    for mic in microphones:
        logging.info(f"- {mic.name.encode('ascii', 'ignore').decode('ascii')}")

    # Set up default audio device
    default_speaker = sc.default_speaker()
    loopback = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
    logging.info(f"\nUsing default speaker: {default_speaker.name}")

    # Start recording thread
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

    try:
        print("Real-time audio transcription and translation started. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping recording...")
        is_recording = False
        recording_thread.join(timeout=2)
        logging.info("Recording stopped.")