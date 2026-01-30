import torch
import torchaudio
import numpy as np
from TTS.config import load_config
from TTS.tts.models import setup_model
import argparse
import os
import json
import time
from tqdm import tqdm
import onnxruntime as ort

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure CUDA provider is available for ONNX Runtime
assert 'CUDAExecutionProvider' in ort.get_available_providers(), "CUDAExecutionProvider is not available! Check your ONNX Runtime and CUDA installation."

if DEVICE != 'cuda':
    print("CUDA not available. Please verify your GPU and driver configuration.")
    exit()

def inference_onnx(model, x, x_lengths=None, speaker_id=None, language_id=None, d_vector=None):
    """Optimized for ONNX inference without unnecessary CPU-GPU transfers."""
    
    # Convert PyTorch tensors to NumPy for ONNX execution
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if x_lengths is None:
        x_lengths = np.array([x.shape[1]], dtype=np.int64)
    elif isinstance(x_lengths, torch.Tensor):
        x_lengths = x_lengths.cpu().numpy()

    scales = np.array(
        [model.inference_noise_scale, model.length_scale, model.inference_noise_scale_dp],
        dtype=np.float32,
    )

    input_params = {
        "input": x,
        "input_lengths": x_lengths,
        "scales": scales,
        "d_vector": d_vector
    }
    
    if speaker_id is not None:
        input_params["sid"] = np.array([speaker_id], dtype=np.int64)

    if language_id is not None:
        input_params["langid"] = np.array([language_id], dtype=np.int64)

    # Execute ONNX inference
    audio = model.onnx_sess.run(["output"], input_params)
    return audio[0][0]

def main():
    parser = argparse.ArgumentParser(description="Run optimized GPU inference on YourTTS ONNX models.")
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the exported .onnx model file.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config.json file.')
    parser.add_argument('--spk_emb', type=str, required=True, 
                        help='Path to the d_vector_speakers.json file.')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the metadata CSV/txt file with sentences.')
    
    # Optional arguments with improved help text
    parser.add_argument('--audio_format', default='.wav', type=str, 
                        help='Output audio format extension (default: .wav).')
    parser.add_argument('--output_folder', default='sentences_inference_time_pt', type=str, 
                        help='Directory to save synthesized audio files.')
    parser.add_argument('--sr', default=22050, type=int, 
                        help='Output sample rate (default: 22050).')
    parser.add_argument('--output_json', default='inference_times.json', type=str, 
                        help='Path to save performance metrics JSON.')
    
    args = parser.parse_args()

    # Read sentences from metadata
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    with open(args.input_file) as f:
        sentences = f.readlines()

    os.makedirs(args.output_folder, exist_ok=True)

    # Setup configuration
    config = load_config(args.config)
    config.model_args["d_vector_file"] = args.spk_emb
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize model and load directly to GPU
    print(f"Loading ONNX model on {DEVICE}...")
    model = setup_model(config).to(DEVICE)
    model.load_onnx(args.checkpoint, cuda=True)

    # Get speaker embedding (ljspeech as reference)
    d_vector = model.speaker_manager.get_mean_embedding("ljspeech")
    d_vector = np.asarray(d_vector, dtype=np.float32)[None, :]

    results = {}

    print(f"Starting GPU-accelerated inference for {len(sentences[:1000])} sentences...")
    for index, line in enumerate(tqdm(sentences[:1000])):
        line = line.strip()
        if not line or "|" not in line:
            continue
            
        parts = line.split("|")
        sentence = parts[1] # Assumes format: id|text|speaker or similar

        # Prepare text input and move to GPU
        text_inputs = torch.tensor(
            model.tokenizer.text_to_ids(sentence, language=None),
            dtype=torch.int64,
        ).to("cuda", non_blocking=True)[None, :]

        # Timing and inference
        start_time = time.time()
        waveform = inference_onnx(model, text_inputs, d_vector=d_vector)
        inference_time = time.time() - start_time

        duration = waveform.size / args.sr
        rtf = inference_time / duration if duration > 0 else float('inf')

        filename = f"output-{index:04d}{args.audio_format}"
        filepath = os.path.join(args.output_folder, filename)

        # Performance Log
        print(f"Inference: {inference_time:.2f}s | RTF: {rtf:.2f} | File: {filename}")

        # Save audio output
        torch_audio = torch.from_numpy(waveform)
        if torch_audio.ndim == 1:
            torch_audio = torch_audio.unsqueeze(0)
        torchaudio.save(filepath, torch_audio, args.sr)

        results[filename] = {
            "inference_time": inference_time,
            "duration": duration,
            "rtf": rtf,
            "text": sentence
        }

    # Export performance results
    with open(args.output_json, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Inference complete. Metrics saved to {args.output_json}")

if __name__ == "__main__":
    main()