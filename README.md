# A RISC-V-based Hybrid MPC–LLM Framework for Adaptive Autonomous Driving

## Race stack installation
For quick deployment, a docker image can be used. The full docker structure and guidelines on how to use both in simulation and in the real platform can be found in the [Docker Guidelines README](./.docker_utils/README.md).

For a quick example of the race stack in action, first build the base docker image with `docker compose`:
```bash
docker compose build base_x86
```

Then export the needed environment variables and build the simulator container:
```bash
export UID=$(id -u)
export GID=$(id -g)
docker compose build sim_x86
```

Then check that the following folder structure is existing:
```bash
<race_stack directory>/../
...
├── race_stack_cache
│   └── noetic
│       ├── build
│       ├── devel
│       └── logs
└ ...
```
It can be created from the command line, for example:
```bash
cd <race_stack folder>
mkdir -p ../race_stack_cache/noetic/build ../race_stack_cache/noetic/devel ../race_stack_cache/noetic/logs
```
Then launch the devcontainer from VS Code

To now test the simulator, launch the simulator with base system with the following command:
```bash
roslaunch stack_master base_system.launch sim:=true map_name:=test_map
```

and then, in a new terminal, launch the timetrials system with the following command:
```bash
roslaunch stack_master time_trials.launch racecar_version:=NUC2
```
For more information on how to run the different modules on the car, refer to the [`stack_master`](./stack_master/README.md) README or to the READMEs in the [checklist](./stack_master/checklists/) directory.


## LLMxRobot installation
1. Build the Docker container (adapt `CUDA_ARCH` accordingly: `86` for RTX 30xx, `89` for 40xx, `61` for 10xx):
```bash
docker build --build-arg CUDA_ARCH=<your_compute_capability> -t embodiedai -f .docker_utils/Dockerfile.cuda .
```

2. Mount the container to the project directory:
```bash
./.docker_utils/main_dock.sh cuda
```

3. Attach to the container (or use VS Code Remote Containers):
```bash
docker exec -it embodiedai_dock /bin/bash
```


## Daily driver commands cheat-sheet
Clone repo with submodules:
```bash
git clone --recurse-submodules git@github.com:<you>/RISCVxLLMxRobot.git
```

Initialize submodules later:
```bash
git submodule update --init --recursive
```

Move read-only submodules forward:
```bash
git submodule update --remote third_party/llama.cpp third_party/voyager-sdk
```

Rebase forks on upstream and bump pointers:
```bash
./scripts/update_all.sh
```

Pull upstream changes for the read-only submodules:
```bash
git submodule update --remote third_party/llama.cpp third_party/voyager-sdk
git add third_party/llama.cpp third_party/voyager-sdk
git commit -m "Bump llama.cpp & voyager-sdk submodules"
```

Pull upstream changes for the forked submodules:
```bash
# race_stack
(
  cd src/race_stack
  git fetch upstream
  # Update your main first (if you keep a clean main)
  git checkout main
  git rebase upstream/main
  git push --force-with-lease origin main

  # Rebase your feature branch(es) on top of the refreshed main
  git checkout feat/my-change
  git rebase main
  git push --force-with-lease
)

# LLMxRobot
(
  cd src/LLMxRobot
  git fetch upstream
  git checkout main
  git rebase upstream/main
  git push --force-with-lease origin main

  git checkout feat/another-change
  git rebase main
  git push --force-with-lease
)

# Back in the superproject: record the new submodule SHAs
git add src/race_stack src/LLMxRobot
git commit -m "Bump race_stack & LLMxRobot to latest rebased commits"
```


## Useful links
- Unsloth models: https://docs.unsloth.ai/get-started/all-our-models#llama-models
- Voyager LLM: https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.4/docs/tutorials/llm.md
- SFT models: https://huggingface.co/nibauman/models?p=0


## How to use with singularity
1. Build Docker image by following the steps detailed in the Forza repository: `https://github.com/ForzaETH/race_stack/tree/main`.

2. Get Docker image name:
```bash
docker ps
```

Which outputs something like this, where the name is `vsc-race_stack-54de3ff3bbfca9751501dc118026c569343e10107f8c8150ba454185f6000277-uid`.
```bash
CONTAINER ID   IMAGE                                                                                 COMMAND                  CREATED        STATUS        PORTS     NAMES
30feed8f30a4   vsc-race_stack-54de3ff3bbfca9751501dc118026c569343e10107f8c8150ba454185f6000277-uid   "/bin/sh -c 'echo Co…"   13 hours ago   Up 13 hours             forzaeth_devcontainer
```

3. Save Docker image as `.tar` file. For example:
```bash
docker save vsc-race_stack-54de3ff3bbfca9751501dc118026c569343e10107f8c8150ba454185f6000277-uid -o race-stack.tar
```

4. Stream to computer with rsync and place it in the `/scratch/$USER` folder. For example:
```bash
rsync -v sem25h27@finsteraarhorn.ee.ethz.ch:/home/sem25h27/race-stack.tar /scratch/sem25h27/
```

5. Build the image as sandbox using singularity:
```bash
singularity build --sandbox race-stack docker-archive://race-stack.tar
```

6. Create a ROS workspace to build the repository:
```bash
cd /scratch/sem25h27/catkin_ws/src
git clone --recurse-submodules https://github.com/ForzaETH/race_stack.git
```

7. Run Singularity image with NVIDIA support:
```bash
cd /scratch/sem25h27
singularity shell --nv race-stack
```

8. Inside the container, head to the catwin workspace:
```bash
source catkin_ws/devel/setup.bash
```

9. Run simulation inside Singularity container:
```bash
roslaunch stack_master base_system.launch sim:=true map_name:=test_map
roslaunch stack_master time_trials.launch racecar_version:=NUC2
```

---

# Inference on the Axelera board:
1. Follow the steps in: https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.4/docs/tutorials/llm.md

2. Source the Axelera venv:
```bash
cd ~/dev/RISCVxLLMxRobot/third_party/voyager-sdk
source venv/bin/activate
```

3. Install LLMxRobot dependencies:
```bash
cd ~/dev/RISCVxLLMxRobot/src/LLMxRobot
pip3 install --upgrade pip setuptools wheel
pip3 install packaging
pip3 install --upgrade --force-reinstall \
    "torch==2.4.*" "torchvision==0.19.*" --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements_voyager.txt
```

4. Call the inference model:
```bash
python3 -m tests.decision_tester.decision_tester --dataset all --ax_local --local_workdir ~/dev/RISCVxLLMxRobot/third_party/voyager-sdk --local_venv ~/dev/RISCVxLLMxRobot/third_party/voyager-sdk/venv/bin/activate --local_run "python3 ~/dev/RISCVxLLMxRobot/third_party/voyager-sdk/inference_llm.py llama-3-2-3b-1024-4core-static"
```

# Usage with local RAG
For online RAG, you can simply use:
```bash
python3 -m tests.decision_tester.decision_tester --model nibauman/RobotxLLM_Qwen7B_SFT --dataset all --rag
```

For offline RAG, you can either:
```bash
# Option A: build on the fly from prompts/RAG_memory.txt (same splitter/chunking)
python3 -m tests.decision_tester.decision_tester --model nibauman/RobotxLLM_Qwen7B_SFT --dataset all --rag --rag_offline

# Option B: use a prebuilt index
python3 -m inference.rag_build_index --corpus_dir prompts --index_path data/rag_index/offline
python3 -m tests.decision_tester.decision_tester --model nibauman/RobotxLLM_Qwen7B_SFT --dataset all --rag --rag_offline --rag_index data/rag_index/offline
```


# Export LLMxRobot docker image for Singularity
1) Build the Docker image for 40-series:
```bash
# Clone
git clone https://github.com/fgarciacardenas/LLMxRobot
cd LLMxRobot

# Build the CUDA image targeting 40xx (sm_89)
docker build \
  --build-arg CUDA_ARCH=89 \
  -t llmxrobot:cuda89 \
  -f .docker_utils/Dockerfile.cuda .
```

Notes:
- You don’t need a GPU to build this; `nvcc` cross-compiles for the target SM.

2) Export the Docker image to a tar file:
```bash
docker save -o llmxrobot-cuda89.tar llmxrobot:cuda89
# (optional) gzip it before transfer
gzip llmxrobot-cuda89.tar
```

3) On the server: build a Singularity/Apptainer .sif from the Docker archive:
```bash
# If you gzipped it, gunzip first
gunzip llmxrobot-cuda89.tar.gz  # if applicable

# Build the SIF from the Docker archive
apptainer build llmxrobot-cuda89.sif docker-archive://llmxrobot-cuda89.tar
```

Notes:
- The `docker-archive://` flow is the documented way to convert a saved Docker image to Apptainer/Singularity without pushing to a registry. 
- If you prefer a registry, push `llmxrobot:cuda89` to Docker Hub/GHCR and do `apptainer build llmxrobot-cuda89.sif docker://user/repo:tag` on the server.

4) Run with GPU access on the server
Use `--nv` so Apptainer injects the host’s NVIDIA driver/libs into the container runtime.

```bash
apptainer exec --nv llmxrobot-cuda89.sif nvidia-smi
```