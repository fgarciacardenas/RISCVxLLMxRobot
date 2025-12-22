# Running the Voyager SDK on Firefly with Axelera Metis AIPU

This README documents the troubleshooting steps and commands needed to get the [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk/) working on a **Firefly RK3588 board** with the **Axelera Metis AIPU**.

---

## 1. Initial Problem

Running inference with Voyager SDK failed:

```bash
./inference_llm.py llama-3-2-3b-1024-4core-static --prompt "Give me a joke"
```

Error output:

```
ERROR   : AXR_ERROR_CONNECTION_ERROR: No AIPU driver found in lsmod output
```

This means the **Axelera AIPU kernel driver was not installed or loaded**.

---

## 2. Verifying Hardware Detection

Before installing drivers, ensure that the board detects the Metis accelerator via PCIe:

```bash
sudo lspci -nn | grep -i -E 'axelera|aipu|processing accelerators|metis' || echo "No Metis in lspci"
uname -a
```

Expected output example:
```
01:00.0 Processing accelerators [1200]: Axelera AI Metis AIPU (rev 02) [1f9d:1100]
```

If you **do not** see this, the hardware is not properly connected or PCIe support is missing.

---

## 3. Install Kernel Headers and DKMS

The Metis driver is built via DKMS, so kernel headers must be installed:

```bash
sudo apt install -y dkms linux-headers-$(uname -r) || echo "Install your Firefly kernel headers package"
```

If headers are missing, install the Firefly-provided kernel headers for your exact kernel version (e.g., `5.10.160`).

---

## 4. Add Axelera APT Repository

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL "https://software.axelera.ai/artifactory/api/security/keypair/axelera/public" | gpg --dearmor | sudo tee /etc/apt/keyrings/axelera.gpg >/dev/null

echo "deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source/ stable main" | sudo tee /etc/apt/sources.list.d/axelera.list >/dev/null

sudo apt update
```

---

## 5. Install Axelera Packages

Initially, the unversioned packages were not found:

```
E: Unable to locate package axelera-runtime
E: Unable to locate package axelera-device
```

The correct packages are **versioned**, so list and install them:

```bash
apt-cache search --names-only axelera | sort
apt-cache policy metis-dkms
sudo apt install -y metis-dkms axelera-runtime-1.4.2 axelera-device-1.4.2
```

---

## 6. Build and Install the DKMS Driver

At this point, DKMS showed:
```
metis/1.2.4: added
```

But it was not built yet. Building failed initially:
```
Error! Your kernel headers for kernel 5.10.160 cannot be found.
```

This happened because `/lib/modules/5.10.160/build` was missing. Fix it by linking the headers:

```bash
sudo mkdir -p /lib/modules/$(uname -r)
sudo ln -sf /usr/src/linux-headers-$(uname -r) /lib/modules/$(uname -r)/build
```

Then rebuild and install:

```bash
sudo dkms remove metis/1.2.4 --all || true
sudo dkms add -m metis -v 1.2.4
sudo dkms build  -m metis -v 1.2.4
sudo dkms install -m metis -v 1.2.4
```

Verify status:

```bash
dkms status | grep -i metis
# Should show: metis/1.2.4, 5.10.160, aarch64: installed
```

---

## 7. Load and Verify the Module

```bash
sudo modprobe metis
lsmod | grep -i metis
```

Expected:
```
metis  57344  0
```

Check logs and device nodes:

```bash
dmesg | grep -i -E 'metis|axelera|aipu' | tail -n 100
ls -l /dev/metis* 2>/dev/null
```

Example output:
```
crw-rw-rw- 1 root axelera 508, 0 Oct 23 09:54 /dev/metis-0:1:0
```

---

## 8. Validate Runtime Detection
```bash
axdevice
# or
axdevice --report
# or
sudo axdevice --pcie-scan
sudo axdevice --refresh
```

If the driver is loaded correctly, the board will appear in the report.

---

## 9. Run Voyager Inference

Finally, rerun inference:

```bash
./inference_llm.py llama-3-2-3b-1024-4core-static --prompt "Give me a joke"
```

At this stage, Voyager should detect the Axelera device and execute successfully.

---

## 10. Summary of Key Fixes

| Issue | Root Cause | Fix |
|--------|-------------|-----|
| `AXR_ERROR_CONNECTION_ERROR: No AIPU driver found` | Driver not installed | Install `metis-dkms`, `axelera-runtime`, `axelera-device` |
| `Error! Your kernel headers ... cannot be found` | Missing `/lib/modules/<ver>/build` | Link to `/usr/src/linux-headers-<ver>` |
| `modprobe: FATAL: Module metis not found` | DKMS module not built | Run `dkms build` and `dkms install` |
| `axdevice: error: unrecognized arguments: --list` | Wrong CLI flag | Use `axdevice` or `axdevice --report` |

---

### Verified Working Environment
- **Board:** Firefly RK3588
- **Kernel:** 5.10.160 (Firefly vendor image)
- **Axelera SDK:** metis-dkms 1.2.4, runtime 1.4.2, device 1.4.2
- **Voyager SDK:** from GitHub (Axelera AI Hub)

---

### References
- Axelera AI Documentation: https://docs.axelera.ai
- Voyager SDK: https://github.com/axelera-ai-hub/voyager-sdk
- Firefly RK3588 Docs: https://wiki.t-firefly.com/en/ROC-RK3588-PC

