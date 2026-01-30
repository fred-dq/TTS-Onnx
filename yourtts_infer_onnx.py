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

#TTS_PATH = "TTS/"
#sys.path.append(TTS_PATH)

#USE_CUDA = torch.cuda.is_available()
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference_onnx(model, x, x_lengths=None, speaker_id=None, language_id=None, d_vector=None):
    """Performs ONNX inference"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoints_onnx/yourtts_checkpoint.onnx', help='Path to the ONNX checkpoint file')
    parser.add_argument('--config', default='./checkpoints_onnx/config.json', help='Path to the config JSON file')
    parser.add_argument('--spk_emb', default='./checkpoints_onnx/d_vector_speakers.json', help='Path to the speaker embeddings JSON file')
    parser.add_argument('--input_file', default='metadata.csv', help='Input file containing sentences')
    parser.add_argument('--audio_format', default='wav', help='Audio output format (e.g., wav)')
    parser.add_argument('--output_folder', default='sentences_inference_time', help='Folder to save output files')
    parser.add_argument('--sr', default=22050, type=int, help='Sample rate')
    parser.add_argument('--output_json', default='inference_times.json', help='File to save inference performance results')
    args = parser.parse_args()

    # Read input sentences
    with open(args.input_file) as f:
        sentences = f.readlines()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    config = load_config(args.config)
    config.model_args["d_vector_file"] = args.spk_emb
    config.model_args["use_speaker_encoder_as_loss"] = False

    # Initialize and load the model
    model = setup_model(config)
    model.load_onnx(args.checkpoint)

    # Get speaker embedding for reference
    d_vector = model.speaker_manager.get_mean_embedding("ljspeech")
    d_vector = np.asarray(d_vector, dtype=np.float32)[None, :]

    results = {}

    # Process sentences and measure performance
    for index, line in enumerate(tqdm(sentences[:1000])):
        line = line.strip()
        _, sentence, _ = line.split("|")

        text_inputs = np.asarray(
            model.tokenizer.text_to_ids(sentence, language=None),
            dtype=np.int64,
        )[None, :]

        start_time = time.time()
        waveform = inference_onnx(model, text_inputs, d_vector=d_vector)        
        inference_time = time.time() - start_time

        duration = waveform.size / args.sr  # Audio duration in seconds
        rtf = inference_time / duration if duration > 0 else float('inf')

        filename = "output-{0:04d}.{1}".format(index, args.audio_format)
        filepath = os.path.join(args.output_folder, filename)
        
        # Log performance metrics
        print("Inference time: {:.2f} s | Duration: {:.2f} s | RTF: {:.2f} | File: {}".format(
            inference_time, duration, rtf, filepath
        ))

        results[filename] = {
            "inference_time": inference_time,
            "duration": duration,
            "rtf": rtf
        }

    # Export results to JSON
    with open(args.output_json, 'w') as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    main()
