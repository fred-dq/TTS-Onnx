import json
from typing import Dict

import torch
import random
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.utils.speakers import load_speaker_mapping

from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.configs.vits_config import VitsConfig

# Load the configuration
def load_json(json_file: str) -> Dict:
    """Loads data from the specified JSON file."""
    try:
        with open(json_file, "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Warning: JSON file '{json_file}' not found. Using default model.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'. Using default model.")
        return None


def load_model_and_config(config_path: str, checkpoint_path: str, speakers_embeddings_path: str):
    """Loads the model and its configuration."""
    config = load_config(config_path)
    config.model_args["d_vector_file"] = speakers_embeddings_path
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize and load the model
    model = setup_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    return model, config


def main():

    model, config = load_model_and_config(
        config_path="./YourTTS-ar_sabrine//config.json",
        checkpoint_path="./YourTTS-ar_sabrine/checkpoint_2052445.pth",
        speakers_embeddings_path="./YourTTS-ar_sabrine/d_vector_speakers.json"
    )
    print("Model loaded successfully!")

    print(model)
    model.eval()
    
    # Export the model to ONNX format
    model.export_onnx(
        "/home/fred/Projetos/DataQueue/TTS-onnx/yourtts_checkpoint.onnx",
        config=config,
    )


if __name__ == "__main__":
    main()
