# LLMxRobot

This document serves for reference to execute a comparison between GPU, GGUF, and Axelera runs.

## 🚀 Installation

First of all, download the repository with all of its submodules. If you haven't done it yet, please run:
```bash
git submodule update --init --recursive
```

### CUDA Platform (e.g., RTX 30xx / 40xx)

1. Build the Docker container (adapt `CUDA_ARCH` accordingly: `86` for RTX 30xx, `89` for 40xx):
   ```bash
   docker build --build-arg CUDA_ARCH=<your_compute_capability> -t embodiedai -f .docker_utils/Dockerfile.cuda .
   ```

2. Mount the container to the project directory:
   ```bash
   ./.docker_utils/main_dock.sh cuda
   ```

3. Attach to the container:
   ```bash
   docker exec -it embodiedai_dock /bin/bash
   ```
   or use VS Code Remote Containers.

---

### Create .env File
Create a `.env` file in the root directory with the following content:
```bash
HUGGINGFACEHUB_API_TOKEN="<your_huggingface_token>"
OPENAI_API_TOKEN="<your_openai_token>"
WANDB_API_KEY="<your_wandb_api_key>" # Optional
```
This is needed for downloading models and using OpenAI APIs which is required if you want to use `gpt-4o` or for using the modules with their RAG embeddings. **Make sure to keep this file private!**


## 📊 DecisionxLLM Evaluation (autonomy stack not required)
All runs needed for the DecisionxLLM evaluation - no RAG, GPT-4o RAG (online), and BAII RAG (offline) - store their results at the `/src/LLMxRobot/tests/logs` location with the corresponding model name and date. The order employed in the script is online RAG, offline RAG, and no RAG. For further details, please consult the LLMxRobot README. The scripts are found in the `/src/LLMxRobot` folder.


### GPU test
This script runs the 3 DecisionxLLM tests - no RAG, GPT-4o RAG (online), and BAII RAG (offline) - on the available local GPU. Note that at least 12GB of VRAM will be required to run inference on the FP16 models. The user can change the model used for inference via the `--model` argument (see script and decision_tester.py file for more information).
```bash
bash run_decision_tests_gpu.sh
```

### GGUF test
This script runs the 3 DecisionxLLM tests - no RAG, GPT-4o RAG (online), and BAII RAG (offline) - on the available local CPU or GPU (depending on llama.cpp configuration) by loading a downloaded GGUF model. Please download the necessary GGUF model and place it in a folder inside the src/LLMxRobot/models folder. The user can change the model used for inference via the `--model` argument (see script and decision_tester.py file for more information).

The models employed for our report can be found at:
 - https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main (we used the `Phi-3-mini-4k-instruct-q4.gguf` file)
 - https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/tree/main (we used the `Llama-3.2-3B-Instruct-Q4_K_M.gguf` file)

```bash
bash run_decision_tests_gguf.sh
```

### Axelera test
This script runs the 3 DecisionxLLM tests - no RAG, GPT-4o RAG (online), and BAII RAG (offline) - on the available local Axelera board. This script assumes the Axelera board is placed in the same computer running the script. Otherwise, please refer to the LLMxRobot README to run the model via SSH. Similarly, the user can change the model used for inference by modifying the `--local_run` argument.
```bash
bash run_decision_tests_axelera.sh
```


## Result comparison
Here is a short comparison between the results obtained on GPU, GGUF, and Axelera. You can find the detailed JSON logs using GPT-4o RAG in the `axelera-logs` folder. Note that there are some minor differences between the log files due to incremental updates we did to the logging capabilities, but no changes were performed on the actual prompts or computation.

GPU test (`axelera-logs/unsloth_Phi-3-mini-4k-instruct_full_onlineRAG/test_log_unsloth_Phi-3-mini-4k-instruct_forward_samples.json`):
```json
{
    "test_case": "Reverse the car",
    "sample_index": 8,
    "rag_mode": "online",
    "rag_k": 5,
    "rag_threshold": 0.0,
    "rag_max_hits": 5,
    "rag_candidate_count": 5,
    "rag_used_count": 5,
    "rag_details": [
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.",
            "score": 0.831438882907809
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 11:\nUnless told otherwise, you should be driving forwards!",
            "score": 0.8135240984085282
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.",
            "score": 0.8005468181951174
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.",
            "score": 0.7966928563642646
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.",
            "score": 0.7913038728227831
        }
    ],
    "rag_candidates": [...],
    "rag_text": "Hint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.\nHint 11:\nUnless told otherwise, you should be driving forwards!\nHint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.\nHint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.\nHint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.\n",
    "prompt": "\n        You are an AI embodied on an autonomous racing car. The human wants to: Reverse the car \n\n        The car is currently on the track, data available is in the Frenet Corrdinate frame, units are in meters and meters per second. \n        The racing line is a minimal curvature trajectory to optimize the lap time.\n        The data has been sampled for 2 seconds in 8 samples.\n        \n        - The car's position along the racing line is given by the s-coordinate: [10, 12, 14, 16, 18, 20, 22, 24]\n\n        \n        - The car's lateral deviation from the racing line is given by the d-coordinate: [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]\n\n        \n        - The car's speed along the racing line is given by the s-speed: [6, 6.5, 6, 5.5, 6, 6, 6.5, 6]\n\n        \n        - The car's speed perpendicular to the racing line is given by the d-speed: [0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1]\n\n        \n        - The distance to the left wall is: [1.5, 1.4, 1.3, 1.6, 1.5, 1.4, 1.5, 1.4]\n\n\n        - The distance to the right wall is: [1.3, 1.5, 1.4, 1.3, 1.5, 1.4, 1.3, 1.5]\n\n \n        - Bool if the car is reversing: False\n\n          \n        - Bool if the car has crashed: False\n\n        \n        - Bool if the car is facing the wall: False\n\n\n   \n        Use these guides to reason: \n\nHint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.\nHint 11:\nUnless told otherwise, you should be driving forwards!\nHint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.\nHint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.\nHint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.\n\n\n    \n        Check if the car is adhering to what the human wants: Reverse the car. Strictly reply in the following format: \n\n        Explanation: <Brief Explanation> \n\n        Adhering to Human: <True/False> \n\n        ",
    "model_response_raw": "Explanation: The car's s-speed is consistently positive and within the range of 5 to 7 m/s, which indicates it is driving forwards. The s-coordinate is increasing, which means the car is moving along the racing line. The d-speed is not close to 0 m/s, so the car is not stopped. The d-coordinate does not oscillate with a magnitude larger than 0.3 metres, so the car is not oscillating around the racing line. The car is not reversing as the s-speed is not negative, and the s-coordinate is increasing.\n\nAdhering to Human: False",
    "sanitized_output": false,
    "structure_followed": true,
    "expected_output": false,
    "prompt_tokens": 718,
    "rag_tokens": 147,
    "output_tokens": 139,
    "model_name": "unsloth_Phi-3-mini-4k-instruct"
}
```

GGUF test (`axelera-logs/microsoft_Phi-3-mini-4k-instruct-gguf_onlineRAG/test_log_models_microsoft_Phi-3-mini-4k-instruct-gguf_forward_samples.json`):
```json
{
    "test_case": "Reverse the car",
    "sample_index": 8,
    "rag_mode": "online",
    "rag_k": 5,
    "rag_details": [
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "content": "Hint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples."
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "content": "Hint 11:\nUnless told otherwise, you should be driving forwards!"
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "content": "Hint 4:\nIf the s-speed is close to 0m/s, then the car is stopped."
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "content": "Hint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s."
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "content": "Hint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line."
        }
    ],
    "rag_text": "Hint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.\nHint 11:\nUnless told otherwise, you should be driving forwards!\nHint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.\nHint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.\nHint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.\n",
    "prompt": "\n        You are an AI embodied on an autonomous racing car. The human wants to: Reverse the car \n\n        The car is currently on the track, data available is in the Frenet Corrdinate frame, units are in meters and meters per second. \n        The racing line is a minimal curvature trajectory to optimize the lap time.\n        The data has been sampled for 2 seconds in 8 samples.\n        \n        - The car's position along the racing line is given by the s-coordinate: [10, 12, 14, 16, 18, 20, 22, 24]\n\n        \n        - The car's lateral deviation from the racing line is given by the d-coordinate: [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]\n\n        \n        - The car's speed along the racing line is given by the s-speed: [6, 6.5, 6, 5.5, 6, 6, 6.5, 6]\n\n        \n        - The car's speed perpendicular to the racing line is given by the d-speed: [0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1]\n\n        \n        - The distance to the left wall is: [1.5, 1.4, 1.3, 1.6, 1.5, 1.4, 1.5, 1.4]\n\n\n        - The distance to the right wall is: [1.3, 1.5, 1.4, 1.3, 1.5, 1.4, 1.3, 1.5]\n\n \n        - Bool if the car is reversing: False\n\n          \n        - Bool if the car has crashed: False\n\n        \n        - Bool if the car is facing the wall: False\n\n\n   \n        Use these guides to reason: \n\nHint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.\nHint 11:\nUnless told otherwise, you should be driving forwards!\nHint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.\nHint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.\nHint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.\n\n\n    \n        Check if the car is adhering to what the human wants: Reverse the car. Strictly reply in the following format: \n\n        Explanation: <Brief Explanation> \n\n        Adhering to Human: <True/False> \n\n        ",
    "model_response_raw": "\n        Explanation: The car is not adhering to the human's request to reverse as the s-speed is positive and the s-coordinate is increasing.\n\n        Adhering to Human: False\n\n        ",
    "sanitized_output": false,
    "expected_output": false,
    "prompt_tokens": 718,
    "rag_tokens": 147,
    "output_tokens": 50
}
```

Axelera board (`axelera-logs/axelera_phi-3_onlineRAG/test_log_local_phi-3_forward_samples.json`):
```json
{
    "test_case": "Reverse the car",
    "sample_index": 8,
    "rag_mode": "online",
    "rag_k": 5,
    "rag_threshold": 0.0,
    "rag_max_hits": 5,
    "rag_candidate_count": 5,
    "rag_used_count": 5,
    "rag_details": [
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.",
            "score": 0.831438882907809
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 11:\nUnless told otherwise, you should be driving forwards!",
            "score": 0.8135240984085282
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.",
            "score": 0.8005468181951174
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.",
            "score": 0.7966928563642646
        },
        {
            "metadata": {
                "source": "prompts/RAG_memory.txt"
            },
            "source": "prompts/RAG_memory.txt",
            "text": "Hint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.",
            "score": 0.7913038728227831
        }
    ],
    "rag_candidates": [...],
    "rag_text": "Hint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.\nHint 11:\nUnless told otherwise, you should be driving forwards!\nHint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.\nHint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.\nHint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.\n",
    "prompt": "\n        You are an AI embodied on an autonomous racing car. The human wants to: Reverse the car \n\n        The car is currently on the track, data available is in the Frenet Corrdinate frame, units are in meters and meters per second. \n        The racing line is a minimal curvature trajectory to optimize the lap time.\n        The data has been sampled for 2 seconds in 8 samples.\n        \n        - The car's position along the racing line is given by the s-coordinate: [10, 12, 14, 16, 18, 20, 22, 24]\n\n        \n        - The car's lateral deviation from the racing line is given by the d-coordinate: [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]\n\n        \n        - The car's speed along the racing line is given by the s-speed: [6, 6.5, 6, 5.5, 6, 6, 6.5, 6]\n\n        \n        - The car's speed perpendicular to the racing line is given by the d-speed: [0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1]\n\n        \n        - The distance to the left wall is: [1.5, 1.4, 1.3, 1.6, 1.5, 1.4, 1.5, 1.4]\n\n\n        - The distance to the right wall is: [1.3, 1.5, 1.4, 1.3, 1.5, 1.4, 1.3, 1.5]\n\n \n        - Bool if the car is reversing: False\n\n          \n        - Bool if the car has crashed: False\n\n        \n        - Bool if the car is facing the wall: False\n\n\n   \n        Use these guides to reason: \n\nHint 5:\nThe car is reversing if the s-speed is negative or if the s-coordinate is reducing over multiple samples.\nHint 11:\nUnless told otherwise, you should be driving forwards!\nHint 4:\nIf the s-speed is close to 0m/s, then the car is stopped.\nHint 2:\nUnless specified differently by the human, the car is usually driving at speeds between 5 and 7m/s.\nHint 7:\nIf the d-coordinate oscillates from positive to negative values with larger magnitude than 0.3 metres, then the car is oscillating around the racing line.\n\n\n    \n        Check if the car is adhering to what the human wants: Reverse the car. Strictly reply in the following format: \n\n        Explanation: <Brief Explanation> \n\n        Adhering to Human: <True/False> \n\n        ",
    "model_response_raw": "The car is oscillating around the racing line because the d-coordinate shows a fluctuation between positive and negative values with a magnitude larger than 0.3m, which indicates that the car is not maintaining a constant distance from the racing line but rather oscillating around it.\nThe car is oscillating around the racing line because the d-coordinate shows a fluctuation between positive and negative values with a magnitude larger than 0.3m, which indicates that the car is not maintaining a constant distance from the racing line but rather oscillating around it.\nThe car is oscillating around the racing line because the d-coordinate shows a fluctuation between positive and negative values with a magnitude larger than 0.3m, which indicates that",
    "sanitized_output": null,
    "expected_output": false,
    "prompt_tokens": 718,
    "rag_tokens": 147,
    "output_tokens": 159
}
```


## 📄 Related papers

**Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models:**
```bibtex
@article{baumann2025enhancing,
  title={Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models},
  author={Baumann, Nicolas and Hu, Cheng and Sivasothilingam, Paviththiren and Qin, Haotong and Xie, Lei and Magno, Michele and Benini, Luca},
  journal={arXiv preprint arXiv:2504.11514},
  year={2025}
}
```
**RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning:**
```bibtex
@misc{boyle2025robotxr1enablingembodiedrobotic,
      title={RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning}, 
      author={Liam Boyle and Nicolas Baumann and Paviththiren Sivasothilingam and Michele Magno and Luca Benini},
      year={2025},
      eprint={2505.03238},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.03238}, 
}
```
