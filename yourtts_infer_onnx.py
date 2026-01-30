import torch
import torchaudio
import numpy as np
from TTS.config import load_config
from TTS.tts.models import setup_model
import argparse
import os
import sys
import json
import time
from tqdm import tqdm

def inference_onnx(model, x, x_lengths=None, speaker_id=None, language_id=None, d_vector=None):
    """ONNX inference execution with input parameter mapping."""

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if x_lengths is None:
        x_lengths = np.array([x.shape[1]], dtype=np.int64)

    if isinstance(x_lengths, torch.Tensor):
        x_lengths = x_lengths.cpu().numpy()
    
    scales = np.array(
        [model.inference_noise_scale, model.length_scale, model.inference_noise_scale_dp],
        dtype=np.float32,
    )
    
    input_params = {"input": x, "input_lengths": x_lengths, "scales": scales, "d_vector": d_vector}
    
    if speaker_id is not None:
        input_params["sid"] = torch.tensor([speaker_id]).cpu().numpy()
    if language_id is not None:
        input_params["langid"] = torch.tensor([language_id]).cpu().numpy()

    audio = model.onnx_sess.run(
        ["output"],
        input_params,
    )
    return audio[0][0]


def main():
    parser = argparse.ArgumentParser(description="Run YourTTS inference using an ONNX model and measure performance.")
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the exported .onnx model file.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config.json file associated with the model.')
    parser.add_argument('--spk_emb', type=str, required=True, 
                        help='Path to the d_vector_speakers.json file containing speaker embeddings.')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the metadata file (e.g., CSV) containing sentences to synthesize.')
    
    # Optional arguments with improved help text
    parser.add_argument('--audio_format', default='wav', type=str,
                        help='Output audio file extension/format (default: wav).')
    parser.add_argument('--output_folder', default='sentences_inference_time', type=str,
                        help='Directory where the generated audio files will be saved.')
    parser.add_argument('--sr', default=22050, type=int, 
                        help='Sample rate for the output audio (default: 22050).')
    parser.add_argument('--output_json', default='inference_times.json', type=str,
                        help='Path to save the JSON file containing performance results (Inference Time, RTF).')
    
    args = parser.parse_args()

    # Load input metadata
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    with open(args.input_file) as f:
        sentences = f.readlines()

    os.makedirs(args.output_folder, exist_ok=True)

    # Configuration setup
    config = load_config(args.config)
    config.model_args["d_vector_file"] = args.spk_emb
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize and load the ONNX model
    print(f"Initializing model from {args.checkpoint}...")
    model = setup_model(config)
    model.load_onnx(args.checkpoint)

    # Get mean embedding (using ljspeech as default reference)
    d_vector = model.speaker_manager.get_mean_embedding("ljspeech")
    d_vector = np.asarray(d_vector, dtype=np.float32)[None, :]

    results = {}

    print("Starting synthesis...")
    for index, line in enumerate(tqdm(sentences[:1000])):
        line = line.strip()
        if not line or "|" not in line:
            continue
            
        parts = line.split("|")
        # Handling different metadata structures (e.g., id|text|speaker)
        sentence = parts[1] if len(parts) > 1 else parts[0]

        text_inputs = np.asarray(
            model.tokenizer.text_to_ids(sentence, language=None),
            dtype=np.int64,
        )[None, :]

        # Performance measurement
        start_time = time.time()
        waveform = inference_onnx(model, text_inputs, d_vector=d_vector)        
        inference_time = time.time() - start_time

        duration = waveform.size / args.sr  # Audio duration in seconds
        rtf = inference_time / duration if duration > 0 else float('inf')

        filename = f"output-{index:04d}.{args.audio_format}"
        filepath = os.path.join(args.output_folder, filename)
        
        # Logging individual performance
        print(f"Inference: {inference_time:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.2f} | File: {filepath}")

        # Optional: Save audio (uncomment if actual wave output is needed)
        # torch_audio = torch.from_numpy(waveform)
        # torchaudio.save(filepath, torch_audio.unsqueeze(0), args.sr)

        results[filename] = {
            "inference_time": inference_time,
            "duration": duration,
            "rtf": rtf,
            "text": sentence
        }

    # Save performance summary
    with open(args.output_json, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"\nInference complete. Performance data saved to: {args.output_json}")

if __name__ == "__main__":
    main()