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
1. Build the Docker container (adapt `CUDA_ARCH` accordingly: `86` for RTX 30xx, `89` for 40xx):
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