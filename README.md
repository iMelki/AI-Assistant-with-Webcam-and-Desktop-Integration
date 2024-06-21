# AI Assistant with Webcam and Desktop Integration

This project is an AI assistant that uses computer vision, speech recognition, and natural language processing to interact with users. It's forked from [svpino/alloy-voice-assistant](https://github.com/svpino/alloy-voice-assistant) with significant modifications and improvements, inspired by a Wes Roth video (https://www.youtube.com/watch?v=_mkyL0Ww_08).

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

2. Create a virtual environment, update pip, and install the required packages:
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

### Notes for Mac users:

Mac users need to install some additional dependencies before installing the Python packages:

1. Install Homebrew if you haven't already:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install portaudio and other dependencies:
   ```
   brew install portaudio
   ```

3. Install PyAudio using pip with specific compilation flags:
   ```
   export LDFLAGS="-L/opt/homebrew/lib"
   export CPPFLAGS="-I/opt/homebrew/include"
   pip install pyaudio
   ```

If you're using Apple Silicon (M1/M2), you might need to use a specific Python version compiled for ARM architecture. Consider using Miniforge to install Python:

```
brew install miniforge
conda create -n myenv python=3.9
conda activate myenv
```

Then follow the installation steps above within this conda environment.

### Notes for Windows users:

Windows users might need to install the following:
1. [Microsoft Visual C++ 14.0 or greater](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. [FFmpeg](https://ffmpeg.org/download.html)

Ensure these are properly installed and added to your system PATH.

## Setting Up API Keys

You need an `OPENAI_API_KEY` and a `GOOGLE_API_KEY` (if using Gemini) to run this code. There are two ways to set these:

1. Using a `.env` file (Recommended):
   Create a `.env` file in the root directory of the project and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. Setting environment variables:
   You can set these directly in your terminal:

   For macOS and Linux:
   ```
   export OPENAI_API_KEY=your_openai_api_key_here
   export GOOGLE_API_KEY=your_google_api_key_here
   ```

   For Windows Command Prompt:
   ```
   set OPENAI_API_KEY=your_openai_api_key_here
   set GOOGLE_API_KEY=your_google_api_key_here
   ```

   For Windows PowerShell:
   ```
   $env:OPENAI_API_KEY="your_openai_api_key_here"
   $env:GOOGLE_API_KEY="your_google_api_key_here"
   ```

   Note: Replace `your_openai_api_key_here` and `your_google_api_key_here` with your actual API keys.

3. Verifying the environment variables:
   After setting the environment variables, you can verify they're set correctly:

   For macOS and Linux:
   ```
   echo $OPENAI_API_KEY
   echo $GOOGLE_API_KEY
   ```

   For Windows Command Prompt:
   ```
   echo %OPENAI_API_KEY%
   echo %GOOGLE_API_KEY%
   ```

   For Windows PowerShell:
   ```
   echo $env:OPENAI_API_KEY
   echo $env:GOOGLE_API_KEY
   ```

## Usage

1. Ensure your virtual environment is activated:
   ```
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

2. Make sure your API keys are set (either in .env file or as environment variables).

3. Run the assistant:
   ```
   python3 assistant.py
   ```

The assistant will start listening for voice input and display the webcam feed or screenshot.

- Speak to interact with the assistant. It will process your speech and provide audio responses.
- Press 's' to switch between webcam and screenshot modes.
- Press 'q' to quit the program.

## Configuration

You can modify the configuration options at the top of the `assistant.py` file to customize the behavior:

- `USE_GEMINI`: Set to `True` to use Google's Gemini model, or `False` to use OpenAI's GPT-4
- `GEMINI_MODEL`: Specify the Gemini model to use
- `GPT4_MODEL`: Specify the GPT-4 model to use
- `WHISPER_MODEL`: Choose the Whisper model for speech recognition
- `TTS_MODEL` and `TTS_VOICE`: Configure the text-to-speech settings

## Troubleshooting

If you encounter issues with API keys, PyAudio, or other dependencies, please check the following:

- Ensure your API keys are correctly set and accessible to the script. Double-check for typos or extra spaces.
- If using environment variables, make sure they are set in the same terminal session where you're running the script.
- If you're still having issues, try restarting your terminal or IDE after setting the environment variables.
- For persistent problems, consider using the `.env` file method instead of environment variables.
- Ensure all system dependencies are correctly installed (portaudio, FFmpeg, etc.)
- Make sure you're using a compatible Python version
- Check that your virtual environment is activated when installing packages and running the script

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original project by [Santiago Valdarrama (svpino)](https://github.com/svpino)
- Inspiration from [Wes Roth's video](https://www.youtube.com/watch?v=_mkyL0Ww_08)
- OpenAI for GPT-4, Whisper, and TTS models
- Google for Gemini model
