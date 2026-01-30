import json
import argparse
import os
from typing import Dict

import torch
from TTS.tts.models import setup_model
from TTS.config import load_config

# VITS imports for configuration consistency
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.configs.vits_config import VitsConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Loads the model and its configuration."""
    config = load_config(config_path)
    config.model_args["d_vector_file"] = speakers_embeddings_path
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize and load the model
    model = setup_model(config)
    
    # Load checkpoint to the appropriate device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Export YourTTS model to ONNX for GPU.")
    
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config.json file')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint.pth')
    parser.add_argument('--spk_emb', type=str, required=True, 
                        help='Path to the d_vector_speakers.json file')
    parser.add_argument('--output', type=str, default='./yourtts_checkpoint_gpu.onnx', 
                        help='Output path for the ONNX file (default: ./yourtts_checkpoint_gpu.onnx)')

    args = parser.parse_args()

    # Load model and config
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model_and_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        speakers_embeddings_path=args.spk_emb
    )
    print("Model loaded successfully!")

    # Prepare for export
    model.eval()
    model.to(device)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting ONNX to: {args.output}...")
    
    # Export using the GPU-specific method
    model.export_onnx_to_gpu(
        args.output,
        config=config,
    )
    
    print("Export completed successfully!")


if __name__ == "__main__":
    main()