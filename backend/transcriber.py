import whisper
import ffmpeg
import os
import tempfile

def transcribe_media(file_path, model_size="base", language=None, output_file=None, verbose=False):
    """Transcribe either audio or video file using ffmpeg-python and Whisper"""
    if verbose:
        print(f"Processing file: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    
    # Check if the file is a video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    is_video = any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    # Create a temporary file for the audio if needed
    temp_audio_file = None
    file_to_transcribe = file_path
    
    
    if is_video:
        if verbose:
            print("Detected video file. Extracting audio...")
        
        # Create a temporary WAV file
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_file.close()
        
        # Extract audio using ffmpeg-python
        try:
            (
                ffmpeg
                .input(file_path)
                .output(temp_audio_file.name, acodec='pcm_s16le', ar=16000, ac=1, vn=None)
                .overwrite_output()
                .run(quiet=not verbose)
            )
            
            file_to_transcribe = temp_audio_file.name
        except ffmpeg.Error as e:
            if temp_audio_file and os.path.exists(temp_audio_file.name):
                os.unlink(temp_audio_file.name)
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
    
    # Load Whisper model and transcribe
    if verbose:
        print(f"Loading Whisper model: {model_size}")
    
    model = whisper.load_model(model_size)
    
    if verbose:
        print("Transcribing...")
    
    # Prepare transcription options
    transcribe_options = {}
    if language:
        transcribe_options["language"] = language
    
    result = model.transcribe(file_to_transcribe, **transcribe_options)
    
    # Clean up temporary file if created
    if temp_audio_file and os.path.exists(temp_audio_file.name):
        if verbose:
            print("Cleaning up temporary files...")
        os.unlink(temp_audio_file.name)
    
    # Output the transcription
    if output_file:
        if verbose:
            print(f"Writing transcript to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result["text"])
    
    return result["text"]