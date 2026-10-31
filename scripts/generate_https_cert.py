"""Generate a local CA and HTTPS certificate for phone camera development."""

import argparse
import datetime as dt
import hashlib
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import PHONE_CAMERA_HOSTNAME  # noqa: E402
from app.services.network_address import get_lan_ipv4  # noqa: E402


CERT_DIR = PROJECT_ROOT / "certs"
PUBLIC_CERT_DIR = PROJECT_ROOT / "frontend" / "certs"


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def generate(force: bool = False) -> None:
    ca_key_path = CERT_DIR / "industrial-safety-ca-key.pem"
    ca_cert_path = CERT_DIR / "industrial-safety-ca.crt"
    server_key_path = CERT_DIR / "industrial-safety-key.pem"
    server_cert_path = CERT_DIR / "industrial-safety-cert.pem"
    phone_ca_path = PUBLIC_CERT_DIR / "industrial-safety-ca.cer"

    outputs = [ca_key_path, ca_cert_path, server_key_path, server_cert_path, phone_ca_path]
    if not force and any(path.exists() for path in outputs):
        existing = ", ".join(str(path) for path in outputs if path.exists())
        raise RuntimeError(f"Certificate files already exist: {existing}. Use --force to replace them.")

    now = dt.datetime.now(dt.timezone.utc)
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Industrial Safety Local Development CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Industrial Safety System"),
    ])
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    lan_ip = get_lan_ipv4()
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    server_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, PHONE_CAMERA_HOSTNAME),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Industrial Safety System"),
    ])
    san_entries = [
        x509.DNSName(PHONE_CAMERA_HOSTNAME),
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address(lan_ip)),
    ]
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    private_format = serialization.PrivateFormat.TraditionalOpenSSL
    encryption = serialization.NoEncryption()
    write_bytes(
        ca_key_path,
        ca_key.private_bytes(serialization.Encoding.PEM, private_format, encryption),
    )
    write_bytes(ca_cert_path, ca_cert.public_bytes(serialization.Encoding.PEM))
    write_bytes(
        server_key_path,
        server_key.private_bytes(serialization.Encoding.PEM, private_format, encryption),
    )
    write_bytes(server_cert_path, server_cert.public_bytes(serialization.Encoding.PEM))
    write_bytes(phone_ca_path, ca_cert.public_bytes(serialization.Encoding.DER))

    fingerprint = hashlib.sha256(ca_cert.public_bytes(serialization.Encoding.DER)).hexdigest().upper()
    grouped = ":".join(fingerprint[index:index + 2] for index in range(0, len(fingerprint), 2))
    print(f"HTTPS certificate generated for {PHONE_CAMERA_HOSTNAME} and {lan_ip}")
    print(f"Server certificate: {server_cert_path}")
    print(f"Server private key: {server_key_path}")
    print(f"Phone CA certificate: {phone_ca_path}")
    print(f"CA SHA-256 fingerprint: {grouped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing certificate files")
    args = parser.parse_args()
    generate(force=args.force)
