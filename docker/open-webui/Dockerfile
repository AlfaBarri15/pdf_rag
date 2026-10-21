FROM ghcr.io/open-webui/open-webui:main

# cryptography 47+ raises SIGILL (exit 132) in Docker Desktop's arm64 VM
# when Open WebUI imports JWT. 46.x still satisfies joserfc (>=45).
RUN pip install --no-cache-dir "cryptography==46.0.3"
