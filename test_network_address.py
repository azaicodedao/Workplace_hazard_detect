import unittest
from unittest.mock import patch

from starlette.requests import Request

from app.api.camera_sessions import _public_base_urls
from app.services.mdns_service import mdns_service
from app.services.network_address import is_loopback_host, replace_url_host


def make_request(scheme="http", host="127.0.0.1", port=8000):
    return Request({
        "type": "http",
        "scheme": scheme,
        "method": "GET",
        "path": "/api/camera-sessions",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", f"{host}:{port}".encode())],
        "client": ("127.0.0.1", 50000),
        "server": (host, port),
    })


class NetworkAddressTests(unittest.TestCase):
    def test_loopback_detection(self):
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertFalse(is_loopback_host("192.168.1.20"))

    def test_replace_url_host_preserves_scheme_and_port(self):
        self.assertEqual(
            replace_url_host("http://127.0.0.1:8000", "192.168.1.20"),
            "http://192.168.1.20:8000",
        )

    @patch("app.api.camera_sessions.get_lan_ipv4", return_value="192.168.1.20")
    def test_http_loopback_uses_detected_lan_ip(self, _mock_ip):
        mdns_service.lan_ip = None
        mdns_service.running = False
        primary, fallback, mode = _public_base_urls(make_request())
        self.assertEqual(primary, "http://192.168.1.20:8000")
        self.assertEqual(fallback, primary)
        self.assertEqual(mode, "auto-lan-ip")

    @patch("app.api.camera_sessions.get_lan_ipv4", return_value="192.168.1.20")
    def test_https_uses_stable_mdns_hostname(self, _mock_ip):
        mdns_service.lan_ip = "192.168.1.20"
        mdns_service.running = True
        try:
            primary, fallback, mode = _public_base_urls(make_request(scheme="https"))
            self.assertEqual(primary, "https://industrial-safety.local:8000")
            self.assertEqual(fallback, "https://192.168.1.20:8000")
            self.assertEqual(mode, "mdns")
        finally:
            mdns_service.running = False
            mdns_service.lan_ip = None


if __name__ == "__main__":
    unittest.main()
