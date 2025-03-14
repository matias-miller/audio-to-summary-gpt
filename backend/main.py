import argparse
import os
import sys
from transcriber import transcribe_media
from llm_providers import create_llm_provider
from dotenv import load_dotenv  # Import python-dotenv

load_dotenv()

def main():
    # Get the current working directory
    current_path = os.getcwd()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe audio/video and generate notes using various LLM APIs")
    
    # Required arguments
    parser.add_argument("file", help="Path to the audio or video file to transcribe", default=current_path)
    
    # Whisper options
    whisper_group = parser.add_argument_group("Whisper Options")
    whisper_group.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large"], default="base",
                      help="Whisper model size to use (default: base)")
    whisper_group.add_argument("--language", help="Language code (e.g., 'en' for English) to use for transcription")
    whisper_group.add_argument("--transcript-output", help="Path to save the transcription output")
    
    # LLM options
    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument("--llm-provider", choices=["openai", "anthropic", "google", "custom"], default="openai",
                     help="LLM provider to use for generating notes (default: openai)")
    llm_group.add_argument("--api-key", help="API key for the LLM provider")
    llm_group.add_argument("--model-name", help="Model name/version to use (provider-specific)")
    llm_group.add_argument("--endpoint", help="Custom API endpoint URL (for custom provider)")
    llm_group.add_argument("--config-file", help="JSON config file for custom provider settings")
    
    # Output options
    parser.add_argument("--notes-output", help="Path to save the generated notes (if not specified, uses filename_notes.md)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--transcript-only", action="store_true", help="Only generate transcript, skip notes generation")
    
    args = parser.parse_args()
    
    try:
        # Perform transcription
        transcript = transcribe_media(
            args.file, 
            model_size=args.whisper_model, 
            language=args.language,
            output_file=args.transcript_output,
            verbose=args.verbose
        )
        
        # If no transcript output is specified and verbose is enabled, print transcript
        if not args.transcript_output and args.verbose:
            print("\nTranscription:")
            print(transcript)
        
        # Generate notes if not in transcript-only mode
        if not args.transcript_only:
            try:
                # Create LLM provider
                llm_provider = create_llm_provider(args)
                
                # Generate notes
                notes = llm_provider.generate_notes(transcript, verbose=args.verbose)
                
                # Determine notes output filename if not specified
                notes_output = args.notes_output
                if not notes_output:
                    base_name = os.path.splitext(args.file)[0]
                    notes_output = f"{base_name}_notes.md"
                
                # Save notes to file
                with open(notes_output, 'w', encoding='utf-8') as f:
                    f.write(notes)
                
                if args.verbose:
                    print(f"Notes saved to: {notes_output}")
                    
            except Exception as e:
                print(f"Error during notes generation: {str(e)}", file=sys.stderr)
                if args.verbose:
                    print("Continuing with transcript only.")
        
        if args.verbose:
            print("Processing complete!")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
