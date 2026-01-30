
# YourTTS ONNX Conversion and Inference Tutorial

This guide provides step-by-step instructions on how to convert a trained YourTTS model to the ONNX format and perform optimized inference.


https://github.com/fred-dq/TTS-Onnx


## 1. Environment Setup

To ensure compatibility with the Coqui TTS libraries and ONNX runtime, it is recommended to use **Python 3.9**.

```bash
# Create a new conda environment
conda create -n tts_onnx python=3.9 pip -y

# Activate the environment
conda activate tts_onnx

# Install pip and required dependencies
cd TTS
pip install -e .
```

If necessary, install the requirements:

```
pip install tts onnx onnxruntime-gpu tqdm
```


---

## 2. Configuration Prep (Crucial Step)

Before starting the conversion, you must update the `config.json` file located in your model's checkpoint directory. The model needs to know where the speaker embeddings are located.

1. Locate your `config.json`.
2. Find the key `"d_vector_file"`.
3. Update the path to point to the `d_vector_speakers.json` file.
> **Note:** This file is generated during training and is typically saved within your training dataset directory.



**Example update in `config.json`:**

```json
{
    ...
    "model_args": {
        "d_vector_file": [
            "PATH_FILE/d_vector_speakers.json"
        ],
        ...
    }
}

```

---

## 3. Model Conversion to ONNX

Use the conversion script to export the PyTorch checkpoint (`.pth`) to the ONNX format. This process will optimize the computation graph for faster execution.

```bash
# Run your conversion script (ensure paths inside the script are correct)
python convert_to_onnx.py

```

Verify that the `.onnx` file has been generated in your specified output directory.

---

## 4. Running Inference

Once you have the ONNX file, you can run optimized inference. The provided inference script is configured to use `CUDAExecutionProvider` for high-performance GPU acceleration.

### Execution

Run the inference script by providing the paths to the ONNX model, the updated config, and your metadata:

```bash
python inference_onnx.py \
    --checkpoint ./path/to/your_model.onnx \
    --config ./path/to/config.json \
    --spk_emb ./path/to/d_vector_speakers.json \
    --input_file ./metadata.csv \
    --output_folder ./results

```

### Key Metrics

The script will output performance metrics for each sentence processed:

* **Inference Time:** Time taken to generate the audio.
* **RTF (Real-Time Factor):** Relationship between processing time and audio duration. If **RTF < 1.0**, the model is faster than real-time.

---

## Troubleshooting

* **CUDA Errors:** Ensure that `onnxruntime-gpu` is installed and that your NVIDIA drivers are compatible with CUDA 11.8+.
* **Path Errors:** Double-check that absolute paths are used in `config.json` for the `d_vector_file` to avoid "File Not Found" errors during model initialization.

```

---

Would you like me to add a section on how to calculate the average RTF across all generated samples at the end of the tutorial?

```