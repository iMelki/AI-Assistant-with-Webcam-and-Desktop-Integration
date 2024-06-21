# AI Assistant with Webcam and Desktop Integration

This project is an AI assistant that uses computer vision, speech recognition, and natural language processing to interact with users. It's forked from [svpino/alloy-voice-assistant](https://github.com/svpino/alloy-voice-assistant) with some modifications inspired by a Wes Roth video (https://www.youtube.com/watch?v=_mkyL0Ww_08)

## Features

- Webcam and screenshot capture modes
- Speech recognition using OpenAI's Whisper
- Natural language processing using either OpenAI's GPT-4 or Google's Gemini
- Text-to-speech responses
- Ambient noise adjustment for better audio recognition

## Prerequisites

- Python 3.7+
- OpenAI API key
- Google API key (if using Gemini)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/wildownes/AI-Assistant-with-Webcam-and-Desktop-Integration
   cd AI-Assistant-with-Webcam-and-Desktop-Integration
   ```

2. Create a virtual environment and activate it:

   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -U pip
   pip install -r requirements.txt
   ```

### Notes for Mac users:

If you're using Apple Silicon, run the following command:

```
brew install portaudio
```

### Notes for Windows users:

Windows users might need to install the following:

1. [Microsoft Visual C++ 14.0 or greater](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. [FFmpeg](https://ffmpeg.org/download.html)

Ensure these are properly installed and added to your system PATH.

4. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Run the assistant:

   ```
   python assistant.py
   ```

2. The assistant will start listening for voice input and display the webcam feed or screenshot.

3. Speak to interact with the assistant. It will process your speech and provide audio responses.

4. Press 's' to switch between webcam and screenshot modes.

5. Press 'q' to quit the program.

## Configuration

You can modify the configuration options at the top of the `assistant.py` file to customize the behavior:

- `USE_GEMINI`: Set to `True` to use Google's Gemini model, or `False` to use OpenAI's GPT-4
- `GEMINI_MODEL`: Specify the Gemini model to use
- `GPT4_MODEL`: Specify the GPT-4 model to use
- `WHISPER_MODEL`: Choose the Whisper model for speech recognition
- `TTS_MODEL` and `TTS_VOICE`: Configure the text-to-speech settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original project by [Santiago Valdarrama (svpino)](https://github.com/svpino)
- OpenAI for GPT-4, Whisper, and TTS models
- Google for Gemini model
