# vLLM Loading Recovery Checklist

This checklist provides steps to ensure reliable vLLM model loading, focusing on snapshot pinning, cache standardization, and offline recovery.

## 1. Environment Variable Standardization

Standardize the HuggingFace cache locations to ensure the server consistently finds the weights.

- [ ] **Set `HF_HOME`**: Define the base directory for HuggingFace data.
  ```bash
  export HF_HOME="/home/hyunwoo/.cache/huggingface"
  ```
- [ ] **Set `HF_HUB_CACHE`**: (Optional) Specifically define the hub cache if it differs from the default `HF_HOME/hub`.
  ```bash
  export HF_HUB_CACHE="$HF_HOME/hub"
  ```
- [ ] **Verify offline mode**: Use `HF_HUB_OFFLINE=1` during production runs to ensure no unexpected downloads occur.
  ```bash
  export HF_HUB_OFFLINE=1
  ```

## 2. Model Path Strategy

Avoid using repository IDs directly in production; use absolute local paths to pinned snapshots.

- [ ] **Use Absolute Paths**: Always provide the full absolute path to the model directory.
  - *Correct*: `/home/hyunwoo/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/<hash>`
  - *Avoid*: `meta-llama/Meta-Llama-3-8B-Instruct` (might trigger network checks/updates)
- [ ] **Pin Snapshot Hash**: If pointing to a cache directory, include the specific snapshot hash folder to ensure reproducibility.

Example command:
```bash
graph-bot llm-server start \
  --model /home/hyunwoo/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
  --port 2427
```

## 3. Pre-download & Recovery

Ensure weights are fully available before starting the server.

- [ ] **Manual Download**: Use `huggingface-cli` to pre-populate the cache.
  ```bash
  huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /path/to/stable/location
  ```
- [ ] **Snapshot Verification**: Check if the `config.json` and weight files (e.g., `*.safetensors`) exist in the target path.
  ```bash
  ls -lh /path/to/model/config.json
  ```
- [ ] **Scripted Recovery**: Use `snapshot_download` from `huggingface_hub` in setup scripts for programmatic pinning.
  ```python
  from huggingface_hub import snapshot_download
  path = snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", revision="8afb486c...")
  ```

## 4. Verification Steps

- [ ] **Check Server Logs**: Monitor `vllm_server.log` for "loading weights" messages.
- [ ] **Check Port**: Verify the server is listening on the expected port.
  ```bash
  netstat -tulnp | grep 2427
  ```
- [ ] **Test Connectivity**: Run a simple curl request to the models endpoint.
  ```bash
  curl http://127.0.0.1:2427/v1/models
  ```
