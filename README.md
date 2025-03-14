# Audio-to-Summary-GPT

## Overview

**Audio-to-Summary-GPT** is a powerful tool designed to convert audio and video content into summarized text using advanced AI models. This repository supports multiple Large Language Models (LLMs), video-to-audio conversion, audio-to-text transcription via Whisper, and note summarization based on the generated transcripts. Whether you're processing podcasts, lectures, or video content, this tool provides a seamless workflow to extract and summarize key information.

---

## Features

- **Multiple LLM Model Support**: Flexibility to choose from various Large Language Models for enhanced performance and customization.
- **Video-to-Audio Conversion**: Extract audio from video files for further processing.
- **Audio-to-Text Transcription**: Utilizes OpenAI's Whisper model for accurate and efficient transcription.
- **Note Summarization**: Generates concise summaries from transcribed text, making it easier to digest large amounts of information.

---

## File Structure
```
audio-to-summary-gpt/
├── backend/ # Backend logic and utilities
├── .gitignore # Specifies files to ignore in version control
├── README.md # This file
├── requirements.txt # List of dependencies
├── env.example # Template for environment variables
├── llm_provider_base.py # Base class for LLM providers
├── llm_providers.py # Implementations of supported LLM providers
├── main.py # Main script to run the application
├── setup.py # Setup script for the project
├── test_main.py # Test cases for the main functionality
└── transcriber.py # Handles audio-to-text transcription
```
---

## Supported LLM Models

The tool currently supports the following LLM models:
- OpenAI GPT
- Anthropic Claude
- Gemini
- xAi
- (Add more as supported)

---

## Dependencies

- Python 3.8+
- OpenAI Whisper
- FFmpeg (for video-to-audio conversion)
- Libraries listed in `requirements.txt`

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

# AudioToSummaryGPT

**AudioToSummaryGPT** is the core functionality of this tool, enabling users to transform audio and video content into summarized text with ease. Leveraging cutting-edge LLM AI models, it ensures accurate transcription and summarization for a variety of use cases.
