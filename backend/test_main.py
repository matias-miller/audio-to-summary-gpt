import unittest
from unittest.mock import patch, MagicMock
import sys
import os

import main


class TestTranscriptionScript(unittest.TestCase):

    @patch("transcriber_script.transcribe_media")
    @patch("transcriber_script.create_llm_provider")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("builtins.open", new_callable=MagicMock)
    def test_main(self, mock_open, mock_parse_args, mock_create_llm_provider, mock_transcribe_media):
        # Prepare test data
        test_file_path = "test_video.mp4"
        mock_args = MagicMock()
        mock_args.file = test_file_path
        mock_args.whisper_model = "base"
        mock_args.language = "en"
        mock_args.transcript_output = "transcript.txt"
        mock_args.llm_provider = "openai"
        mock_args.api_key = "test_api_key"
        mock_args.model_name = "gpt-3"
        mock_args.endpoint = None
        mock_args.config_file = None
        mock_args.notes_output = None
        mock_args.verbose = True
        mock_args.transcript_only = False

        # Mock the return values
        mock_parse_args.return_value = mock_args
        mock_transcribe_media.return_value = "Transcribed text"
        
        # Mock the LLM provider and the generate_notes function
        mock_llm = MagicMock()
        mock_llm.generate_notes.return_value = "Generated notes"
        mock_create_llm_provider.return_value = mock_llm

        # Run the main function
        with patch("sys.stdout", new_callable=MagicMock()) as mock_stdout:
            exit_code = main()

        # Check the correct output and interactions
        mock_transcribe_media.assert_called_once_with(
            test_file_path,
            model_size="base",
            language="en",
            output_file="transcript.txt",
            verbose=True
        )

        mock_create_llm_provider.assert_called_once_with(mock_args)
        mock_llm.generate_notes.assert_called_once_with("Transcribed text", verbose=True)

        mock_open.assert_called_once_with("transcript.txt", 'w', encoding='utf-8')
        mock_stdout.write.assert_called()
        
        self.assertEqual(exit_code, 0)  # Check if the exit code was 0 (success)
        mock_stdout.write.assert_any_call("Transcription completed successfully!")

    @patch("argparse.ArgumentParser.parse_args")
    def test_main_with_error(self, mock_parse_args):
        # Simulate an error scenario
        mock_args = MagicMock()
        mock_args.file = "non_existing_file.mp4"
        mock_parse_args.return_value = mock_args

        with patch("sys.stderr", new_callable=MagicMock()) as mock_stderr:
            exit_code = main()

        # Check if the error is logged to stderr
        mock_stderr.write.assert_called()
        self.assertEqual(exit_code, 1)  # Check if the exit code was 1 (error)


if __name__ == "__main__":
    unittest.main()
