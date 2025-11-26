# LLMxRobot

This document serves for reference to execute a comparison between GPU, GGUF, and Axelera runs.

## 🚀 Installation

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
All runs needed for the DecisionxLLM evaluation - no RAG, GPT-4o RAG (online), and BAII RAG (offline) - store their results at the /src/LLMxRobot/tests/logs location with the corresponding model name and date. The order employed in the script is online RAG, offline RAG, and no RAG. For further details, please consult the LLMxRobot README.


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
