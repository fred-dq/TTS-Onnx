import json
import argparse
import os
from typing import Dict

import torch
from TTS.tts.models import setup_model
from TTS.config import load_config

# Import specific VITS configurations if needed for the environment
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.configs.vits_config import VitsConfig

def load_json(json_file: str) -> Dict:
    """Loads data from the specified JSON file."""
    try:
        with open(json_file, "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Warning: JSON file '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'.")
        return None


def load_model_and_config(config_path: str, checkpoint_path: str, speakers_embeddings_path: str):
    """Loads the model and its configuration from paths."""
    config = load_config(config_path)
    config.model_args["d_vector_file"] = speakers_embeddings_path
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize and load the model
    model = setup_model(config)
    # Using 'cpu' for loading before export is safer, but 'cuda' works if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Convert YourTTS PyTorch checkpoint to ONNX format.")
    
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the model config.json')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the PyTorch checkpoint (.pth)')
    parser.add_argument('--spk_emb', type=str, required=True, 
                        help='Path to the d_vector_speakers.json file')
    parser.add_argument('--output', type=str, default='model.onnx', 
                        help='Output path for the generated ONNX file (default: model.onnx)')

    args = parser.parse_args()

    # Load model and config using provided arguments
    print(f"Loading model from: {args.checkpoint}...")
    model, config = load_model_and_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        speakers_embeddings_path=args.spk_emb
    )
    print("Model loaded successfully!")

    # Set to evaluation mode
    model.eval()

    # Export to ONNX
    print(f"Exporting model to ONNX at: {args.output}...")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model.export_onnx(
        args.output,
        config=config,
    )
    print("Export completed successfully!")


if __name__ == "__main__":
    main()