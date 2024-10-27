import base64
from threading import Lock, Thread
import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
import speech_recognition as sr
import numpy as np
from PIL import ImageGrab
import io

# Load environment variables from a .env file
load_dotenv()

# Configuration options
USE_GEMINI = False  # Set to False to use GPT-4
GEMINI_MODEL = "gemini-1.5-pro"
GPT4_MODEL = "gpt-4o"
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = "english"
TTS_MODEL = "tts-1"
TTS_VOICE = "fable"
USE_WEBCAM = True
WEBCAM_INDEX = 0

# Audio configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
AUDIO_CHANNELS = 1
AUDIO_FORMAT = paInt16

# WebcamAudio configuration
SILENCE_THRESHOLD_MULTIPLIER = 1.2
MAX_AUDIO_DURATION = 10  # seconds
SILENCE_DURATION = 1  # second
MAX_SILENT_CHUNKS = 10

# System prompt for the AI assistant
SYSTEM_PROMPT = """
You are a witty assistant that will use the chat history and the image 
provided by the user to answer its questions.

Use few words on your answers. Go straight to the point. Do not use any
emoticons or emojis. Do not ask the user any questions.

Be friendly and helpful. Show some personality. Do not be too formal.

If the user need a code fix, provide the code fix.

Keep replies short unless specifically needed.
"""

class ScreenshotCapture:
    def capture(self):
        screenshot = ImageGrab.grab()
        buffer = io.BytesIO()
        screenshot.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue())

class WebcamStream:
    def __init__(self):
        try:
            self.stream = VideoCapture(index=WEBCAM_INDEX)
            _, self.frame = self.stream.read()
            self.running = self.frame is not None  # Check if webcam capture was successful
            self.lock = Lock()
        except Exception as e:
            print(f"Warning: Webcam not accessible. Error: {e}")
            self.running = False

    def start(self):
        if not self.running:
            print("Webcam is disabled due to initialization failure.")
            return self
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            if frame is None:
                self.running = False
                print("Warning: Webcam stream stopped unexpectedly.")
                break
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        if not self.running:
            return None
        with self.lock:
            frame = self.frame.copy()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, "stream"):
            self.stream.release()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()

class WebcamAudio:
    def __init__(self):
        try:
            self.audio = PyAudio()
            self.audio_stream = self.audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, 
                                                rate=AUDIO_SAMPLE_RATE, input=True, 
                                                frames_per_buffer=AUDIO_CHUNK_SIZE)
            self.recognizer = sr.Recognizer()
            self.silence_threshold = 300
        except Exception as e:
            print(f"Warning: Audio device not accessible. Error: {e}")
            self.audio_stream = None

    def read_audio(self, max_duration=MAX_AUDIO_DURATION):
        if not self.audio_stream:
            print("Audio device is unavailable.")
            return None
        frames = []
        silent_chunks = 0

        for _ in range(0, int(AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE * max_duration)):
            data = self.audio_stream.read(AUDIO_CHUNK_SIZE)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio_data).mean() < self.silence_threshold:
                silent_chunks += 1
                if silent_chunks > MAX_SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0

        audio_data = b''.join(frames)
        return sr.AudioData(audio_data, AUDIO_SAMPLE_RATE, 2) if frames else None

    def adjust_for_ambient_noise(self, duration=SILENCE_DURATION):
        if not self.audio_stream:
            print("Audio device is unavailable. Skipping adjust for ambient noise.")
        else:
            try:
                print("Adjusting for ambient noise. Please remain silent...")
                frames = []
                for _ in range(0, int(AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE * duration)):
                    data = self.audio_stream.read(AUDIO_CHUNK_SIZE)
                    frames.append(np.frombuffer(data, dtype=np.int16))

                ambient_noise = np.concatenate(frames)
                self.silence_threshold = int(np.abs(ambient_noise).mean() * SILENCE_THRESHOLD_MULTIPLIER)
                print(f"Ambient noise threshold set to: {self.silence_threshold}")
            except Exception as e:
                print(f"Error adjusting for ambient noise: {e}")

    def stop(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, "audio"):
            self.audio.terminate()

class Assistant:
    def __init__(self, model):
        self.model = model
        self.chain = self._create_inference_chain(model)
        self.use_webcam = USE_WEBCAM
        self.capture_device = WebcamStream().start()
        self.screenshot_capture = ScreenshotCapture()
        self.audio_device = WebcamAudio()
        self.recognizer = sr.Recognizer()

    def answer(self, prompt):
        if not prompt:
            return

        print("Prompt:", prompt)

        if self.use_webcam:
            image = self.capture_device.read(encode=True)
        else:
            image = self.screenshot_capture.capture()

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

    def transcribe_audio(self, audio):
        try:
            return self.recognizer.recognize_whisper(audio, model=WHISPER_MODEL, language=WHISPER_LANGUAGE)
        except sr.UnknownValueError:
            print("Whisper could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Whisper service; {e}")
        return None

    def toggle_mode(self):
        self.use_webcam = not self.use_webcam
        print(f"Switched to {'webcam' if self.use_webcam else 'screenshot'} mode")

# Initialize the language model based on configuration
if USE_GEMINI:
    model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
else:
    model = ChatOpenAI(model=GPT4_MODEL)

# Initialize AI assistant
assistant = Assistant(model)

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model=WHISPER_MODEL, language=WHISPER_LANGUAGE)
        assistant.answer(prompt)
    except sr.UnknownValueError:
        print("There was an error processing the audio.")

# Function to initialize webcam
def initialize_camera():
    try:
        cap = cv2.VideoCapture(index=WEBCAM_INDEX)
        if not cap.isOpened():
            raise cv2.error("Camera index out of range or not accessible")
        return cap
    except Exception as e:
        print(f"Webcam initialization failed: {e}")
        return None  # or return a default/fallback

# Function to initialize microphone
def initialize_microphone():
    try:
        microphone = sr.Microphone()
        return microphone
    except Exception as e:
        print(f"Audio initialization failed: {e}")
        return None

# Initialize devices with error handling
camera = initialize_camera()
microphone = initialize_microphone()

# Main loop
if microphone:
    with microphone as source:
        recognizer = sr.Recognizer()
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    print("Press 'q' to quit, 's' to switch modes")
    while True:
        if assistant.use_webcam:
            frame = assistant.capture_device.read()
            cv2.imshow("Capture", frame)
        else:
            screenshot = ImageGrab.grab()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            # Scale the screenshot to 1/8 size
            height, width = screenshot_bgr.shape[:2]
            new_height = int(height * 0.33)
            new_width = int(width * 0.33)
            resized_screenshot = cv2.resize(screenshot_bgr, (new_width, new_height))
            cv2.imshow("Capture", resized_screenshot)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            assistant.toggle_mode()

    # Clean up
    stop_listening(wait_for_stop=False)
    assistant.capture_device.stop()
    assistant.audio_device.stop()
    cv2.destroyAllWindows()
else:
    print("Microphone initialization failed. Exiting...")
