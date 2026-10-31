import os
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_key = project_root / "certs" / "industrial-safety-key.pem"
    default_cert = project_root / "certs" / "industrial-safety-cert.pem"
    ssl_keyfile = os.getenv("SSL_KEYFILE") or (str(default_key) if default_key.exists() else None)
    ssl_certfile = os.getenv("SSL_CERTFILE") or (str(default_cert) if default_cert.exists() else None)
    scheme = "https" if ssl_keyfile and ssl_certfile else "http"
    print(f"Starting Industrial Safety System on {scheme}://0.0.0.0:{os.getenv('PORT', '8000')}")
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
