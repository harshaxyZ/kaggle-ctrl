# kaggle ctrl

A self-hosted control panel and OpenAI-compatible gateway for your own Kaggle GPU notebooks. Add your own credentials, choose your own GGUF model and use one local gateway URL. No accounts, model choices or access keys are included.

## Requirements

- Python 3.11 or newer.
- A Kaggle account that can create GPU notebooks with internet access, plus its API token and username.
- Enough accelerator quota and memory for the model you choose. Queues, session limits and quota resets are controlled by Kaggle. Continuous availability is not guaranteed.
- A GGUF model you are permitted to download and use. Gated/private model downloads are not configured by this template.

## Quick start

Clone this repository, then open a terminal in its directory:

```sh
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python setup_local.py
python run.py
```

Setup prompts for your Kaggle username, token (hidden while typed), model repository, GGUF filename and a local alias. It writes only ignored local files. It refuses to overwrite an existing installation. Do not paste credentials into issues, screenshots or commits.

Open <http://127.0.0.1:8787>. Create your first gateway key locally, then keep it private. The dashboard and inference gateway use keys different from your Kaggle token. Add further accounts only if you own or are authorized to use them, and follow the service's rules.

Select your configured model and account, then start a notebook from the dashboard. Loading can take several minutes and consumes quota. Wait for its ready endpoint before testing. Use Stop to end the notebook when finished. Closing the local dashboard does not guarantee the remote notebook has stopped; verify the notebook on Kaggle too.

## Local configuration

`settings.example.json` contains deliberately fake placeholders, not a usable model selection. Setup writes `.local/settings.json`; review its context size and optional server arguments for your hardware before starting a notebook.

- `default_model`: a key from your `models` list.
- `hf_repo`, `filename`, `revision`: your chosen model source and checkpoint.
- `ctx`, `n_parallel`: context size and parallel slots. Larger values need more memory.
- `mmproj`: optional projector filename from the same repository.
- `server_args`: optional extra llama-server arguments. Unsupported flags can prevent startup.
- `session_seconds`: a local maximum runtime, including startup. It does not override vendor limits.

You can select another config path with `KGCTRL_CONFIG`. `.env` is loaded on startup without overriding variables already set in your shell. See `.env.example`. Keep `HOST=127.0.0.1` unless you understand the security implications.

The generic notebook downloads your selected weights, installs llama-server, waits for readiness and opens a temporary inference tunnel. The local gateway process must keep running to route requests. Tunnel and model-server failures can interrupt service. No stable remote hosting, paid service or external gateway account is provisioned.

## Use the gateway

Copy the base URL and a generated key from the dashboard into your OpenAI-compatible client. Choose the alias you configured. `GET /v1/models` lists live deployments; `POST /v1/chat/completions` supports regular and streaming requests. An optional Anthropic protocol adapter is included; it is not required for the core dashboard.

Useful local control routes:

| Method and path | Action |
| --- | --- |
| `GET /api/state` | Dashboard state |
| `POST /api/generate-key` | Create a gateway key |
| `POST /api/accounts` | Add an authorized account |
| `GET /api/models` | Configured model choices |
| `POST /api/server/start` | Start a notebook |
| `GET /api/deployments` | Inspect notebook state |
| `POST /api/server/stop/{deployment_id}` | Stop one exact notebook |

After bootstrap, network requests need `Authorization: Bearer YOUR_GATEWAY_KEY`. Loopback control calls are trusted by default, so do not share your machine with untrusted processes. The router always requires a valid key.

## Security and private files

This is a local, single-operator tool, not a public multi-tenant hosting service. Initial key bootstrap permits local setup before keys exist. Create the first key before exposing any port. Do not put the dashboard behind a proxy that makes untrusted callers look like localhost. Use HTTPS and access controls for any remote deployment.

The source allowlist in `.gitignore` excludes runtime configuration by default. Never force-add `.env`, `.local/`, account databases, API-key files, generated notebooks, logs, cookies or endpoint state. Kaggle credentials and backend keys are stored locally, so secure the machine and its backups. Generated notebook code contains its backend key and must remain private. See [SECURITY.md](SECURITY.md).

## Tests

```sh
python test_template.py
python test_installer.py
```

The template test uses temporary directories and fake credentials. It checks first-run setup, config loading, generated Python syntax and fail-closed missing configuration. It does not spend quota or validate your chosen model on a GPU. A real notebook test is still required for your model and hardware.

## Project files

- `setup_local.py`, `settings.py`: private first-run configuration.
- `kernel_template.py`, `llama_installer.py`: generic notebook generation and server installation.
- `app.py`, `static/index.html`: local dashboard.
- `server_manager.py`, `accounts.py`: notebook and account lifecycle.
- `router.py`, `keys.py`: request routing and gateway-key management.

## License

MIT, see [LICENSE](LICENSE). Model weights and hosted services have their own terms and licenses. This project does not grant rights to them.
