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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the configuration
def load_json(json_file: str) -> Dict:
    """Loads data from the specified JSON file."""
    try:
        with open(json_file, "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Warning: json file '{json_file}' not found. Using default model.")
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
    config_path="./YourTTS_saud_english_v0.2/config.json"
    checkpoint_path="./YourTTS_saud_english_v0.2/checkpoint.pth"
    speakers_embeddings_path="./YourTTS_saud_english_v0.2/d_vector_speakers.json"
    model, config = load_model_and_config(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        speakers_embeddings_path=speakers_embeddings_path
    )
    print("Model loaded successfully!")
 
    model.eval()

    model.to(device)

    model.export_onnx_to_gpu(
        "./yourtts_checkpoint_gpu.onnx",
        config=config,
    )


if __name__ == "__main__":
    main()
