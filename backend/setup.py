from setuptools import setup, find_packages

setup(
    name="whisper-notes",
    version="0.1.0",
    description="Convert audio/video to transcriptions and notes using Whisper and LLMs",
    author="Developer",
    packages=find_packages(),
    install_requires=[
        "whisper",
        "ffmpeg-python",
        "requests",
    ],
    extras_require={
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "google": ["google-generativeai"],
        "all": ["openai", "anthropic", "google-generativeai"],
    },
    entry_points={
        "console_scripts": [
            "whisper-notes=main:main",
        ],
    },
    python_requires=">=3.7",
)