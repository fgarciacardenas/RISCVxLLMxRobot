# A RISC-V-based Hybrid MPC–LLM Framework for Adaptive Autonomous Driving



### Daily driver commands cheat-sheet
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
