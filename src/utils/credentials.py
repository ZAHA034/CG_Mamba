"""Credential loader for CG-Mamba data pipeline.

Loads API tokens and keys from `config/credentials.json` (gitignored).
Never logs token values. Raises clear errors if a required token is missing.

Usage:
    from src.utils.credentials import get_credential, has_credential

    if has_credential("noaa_ncei_cdo_token"):
        token = get_credential("noaa_ncei_cdo_token")
        # use token for API call
    else:
        # fallback to bulk HTTPS download (no auth)
        pass

Security:
- Tokens are loaded once and cached in module memory
- Never printed, logged, or written to any output
- Missing or placeholder values raise CredentialError with the issuance URL
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS_PATH = REPO_ROOT / "config" / "credentials.json"
TEMPLATE_PATH = REPO_ROOT / "config" / "credentials.template.json"

# Placeholder strings that indicate the user hasn't filled in the token yet
_PLACEHOLDER_PREFIXES = ("PASTE_", "YOUR_", "<", "TODO")


class CredentialError(RuntimeError):
    """Raised when a required credential is missing or unfilled."""


@lru_cache(maxsize=1)
def _load_credentials() -> dict:
    """Load credentials.json once and cache in module memory."""
    if not CREDENTIALS_PATH.exists():
        raise CredentialError(
            f"credentials.json not found at {CREDENTIALS_PATH}.\n"
            f"Copy the template and fill in tokens:\n"
            f"  cp {TEMPLATE_PATH} {CREDENTIALS_PATH}\n"
            f"  # then edit {CREDENTIALS_PATH} with your tokens\n"
            f"See docs/CREDENTIALS.md for token issuance procedures."
        )
    with open(CREDENTIALS_PATH) as f:
        return json.load(f)


def _is_placeholder(value: str | None) -> bool:
    """True if the value is None/empty or still a template placeholder."""
    if value is None or value == "":
        return True
    if isinstance(value, str) and value.startswith(_PLACEHOLDER_PREFIXES):
        return True
    return False


def has_credential(name: str) -> bool:
    """Return True if `name` has a non-placeholder value in credentials.json.

    Use this to conditionally enable code paths that require a token.
    """
    try:
        creds = _load_credentials()
    except CredentialError:
        return False
    return not _is_placeholder(creds.get(name))


def get_credential(name: str, *, required: bool = True) -> str | None:
    """Return the credential value for `name`.

    Args:
        name: Key in credentials.json (e.g., 'noaa_ncei_cdo_token').
        required: If True, raise CredentialError when missing. If False, return None.

    Returns:
        The credential string, or None if required=False and missing.

    Raises:
        CredentialError: If required=True and the credential is missing or placeholder.
    """
    creds = _load_credentials()
    value = creds.get(name)

    if _is_placeholder(value):
        if not required:
            return None
        metadata = creds.get("_token_metadata", {}).get(name, {})
        signup = metadata.get("signup_url", "(see docs/CREDENTIALS.md)")
        purpose = metadata.get("purpose", "(unknown)")
        raise CredentialError(
            f"Credential '{name}' is missing or unfilled in {CREDENTIALS_PATH}.\n"
            f"  Purpose: {purpose}\n"
            f"  Sign up: {signup}\n"
            f"After obtaining, paste it into the '{name}' field in credentials.json."
        )
    return value


def list_required_credentials() -> dict[str, dict]:
    """Return metadata dict for all credentials defined in the template.

    Useful for printing a setup checklist or generating docs.
    """
    if not TEMPLATE_PATH.exists():
        return {}
    with open(TEMPLATE_PATH) as f:
        template = json.load(f)
    return template.get("_token_metadata", {})


if __name__ == "__main__":
    # Quick status check (no token values printed)
    print(f"Credentials file: {CREDENTIALS_PATH}")
    print(f"  exists: {CREDENTIALS_PATH.exists()}")
    if CREDENTIALS_PATH.exists():
        try:
            creds = _load_credentials()
            print("\nStatus per credential (✓ = filled, ✗ = placeholder/missing):")
            for key in [
                "noaa_ncei_cdo_token",
                "census_api_key",
                "delphi_epidata_api_key",
                "github_pat",
            ]:
                status = "✓" if has_credential(key) else "✗"
                print(f"  {status} {key}")
        except Exception as e:
            print(f"  error: {e}")
    else:
        print(f"\nTo set up:")
        print(f"  cp config/credentials.template.json config/credentials.json")
        print(f"  # then edit config/credentials.json with your tokens")
        print(f"\nSee docs/CREDENTIALS.md for issuance procedures.")
