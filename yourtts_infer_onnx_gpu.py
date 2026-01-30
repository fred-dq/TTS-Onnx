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

assert 'CUDAExecutionProvider' in ort.get_available_providers(), "CUDAExecutionProvider is not available!"

if DEVICE != 'cuda':
    print("CUDA not available. Please check your GPU configuration.")
    exit()

def inference_onnx(model, x, x_lengths=None, speaker_id=None, language_id=None, d_vector=None):
    """Optimized to perform ONNX model inference without unnecessary CPU-GPU transfers"""
    
    # Transfer tensors to CUDA and keep as PyTorch tensors
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

    # Run inference
    audio = model.onnx_sess.run(["output"], input_params)
    return audio[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./YourTTS-Ermis-Onnx/yourtts_checkpoint.onnx', help='ONNX checkpoint file path')
    parser.add_argument('--config', default='./YourTTS-Ermis-Onnx/config.json', help='Config JSON file path')
    parser.add_argument('--spk_emb', default='./YourTTS-Ermis-Onnx/d_vector_speakers.json', help='Speaker embeddings JSON file path')
    parser.add_argument('--input_file', default='train.csv', help='Input metadata file')    
    parser.add_argument('--audio_format', default='.wav', help='Output audio format')
    parser.add_argument('--output_folder', default='sentences_inference_time_pt', help='Output folder path')
    parser.add_argument('--sr', default=22050, type=int, help='Sample rate')
    parser.add_argument('--output_json', default='inference_times.json', help='Output JSON for inference performance metrics')
    args = parser.parse_args()

    with open(args.input_file) as f:
        sentences = f.readlines()

    os.makedirs(args.output_folder, exist_ok=True)

    config = load_config(args.config)
    config.model_args["d_vector_file"] = args.spk_emb
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize model and load it onto the GPU
    model = setup_model(config).to(DEVICE)
    model.load_onnx(args.checkpoint, cuda=True)

    d_vector = model.speaker_manager.get_mean_embedding("ljspeech")
    d_vector = np.asarray(d_vector, dtype=np.float32)[None, :]

    results = {}

    # Optimized inference process
    for index, line in enumerate(tqdm(sentences[:1000])):
        line = line.strip()
        _, sentence, _ = line.split("|")

        text_inputs = torch.tensor(
            model.tokenizer.text_to_ids(sentence, language=None),
            dtype=torch.int64,
        ).to("cuda", non_blocking=True)[None, :]  # Send to GPU

        start_time = time.time()
        waveform = inference_onnx(model, text_inputs, d_vector=d_vector)
        inference_time = time.time() - start_time

        duration = waveform.size / args.sr  # Duration in seconds
        rtf = inference_time / duration if duration > 0 else float('inf')

        filename = f"output-{index:04d}{args.audio_format}"
        filepath = os.path.join(args.output_folder, filename)

        print(f"Inference: {inference_time:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.2f} | File: {filepath}")

        torch_audio = torch.from_numpy(waveform)
        torchaudio.save(filepath, torch_audio, args.sr)

        results[filename] = {
            "inference_time": inference_time,
            "duration": duration,
            "rtf": rtf
        }

    # Save results to JSON
    with open(args.output_json, 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()
