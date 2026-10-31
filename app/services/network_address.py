"""Resolve a phone-reachable address without hard-coding the current Wi-Fi IP."""

import ipaddress
import socket
from urllib.parse import urlsplit


LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}


def is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower().strip("[]")
    if normalized in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def get_lan_ipv4() -> str:
    """Return the IPv4 address selected by the OS default network route."""
    probes = (("1.1.1.1", 80), ("8.8.8.8", 80))
    for target in probes:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(target)
            candidate = sock.getsockname()[0]
            if candidate and not is_loopback_host(candidate):
                return candidate
        except OSError:
            pass
        finally:
            sock.close()

    candidates = socket.gethostbyname_ex(socket.gethostname())[2]
    private_candidates = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4 and not address.is_loopback:
            if address.is_private:
                private_candidates.append(candidate)
            else:
                return candidate
    if private_candidates:
        return private_candidates[0]
    raise RuntimeError("Could not detect a phone-reachable LAN IPv4 address")


def replace_url_host(base_url: str, host: str) -> str:
    parsed = urlsplit(base_url)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"
