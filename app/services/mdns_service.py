"""Advertise a stable .local hostname for phone camera pairing."""

import socket

from app.config import APP_PORT, ENABLE_MDNS, PHONE_CAMERA_HOSTNAME
from app.services.network_address import get_lan_ipv4


class MdnsService:
    def __init__(self):
        self.zeroconf = None
        self.info = None
        self.running = False
        self.lan_ip = None
        self.last_error = None

    def start(self) -> None:
        if not ENABLE_MDNS or self.running:
            return
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf

            self.lan_ip = get_lan_ipv4()
            hostname = PHONE_CAMERA_HOSTNAME.rstrip(".") + "."
            self.info = ServiceInfo(
                "_http._tcp.local.",
                "Industrial Safety System._http._tcp.local.",
                addresses=[socket.inet_aton(self.lan_ip)],
                port=APP_PORT,
                properties={"path": "/", "camera": "phone-pairing"},
                server=hostname,
            )
            self.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
            self.zeroconf.register_service(
                self.info,
                allow_name_change=True,
                cooperating_responders=True,
            )
            self.running = True
            self.last_error = None
            print(f"[mDNS] {PHONE_CAMERA_HOSTNAME} -> {self.lan_ip}:{APP_PORT}")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc!r}"
            self.stop()
            print(f"[mDNS] Disabled: {exc}")

    def refresh_network(self) -> None:
        """Re-register the hostname when DHCP/network selection changes."""
        if not ENABLE_MDNS:
            return
        current_ip = get_lan_ipv4()
        if self.running and self.lan_ip == current_ip:
            return
        self.stop()
        self.start()

    def stop(self) -> None:
        if self.zeroconf and self.info:
            try:
                self.zeroconf.unregister_service(self.info)
            except Exception:
                pass
        if self.zeroconf:
            try:
                self.zeroconf.close()
            except Exception:
                pass
        self.zeroconf = None
        self.info = None
        self.running = False


mdns_service = MdnsService()
