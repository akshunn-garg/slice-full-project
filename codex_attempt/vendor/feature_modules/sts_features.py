import re
import math
import email
import socket
import ipaddress
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
from datetime import datetime, timezone

import requests


# ============================================================
# CONFIG
# ============================================================

NEW_DOMAIN_LABELS = {
    0: 0,       # <= 7 days
    1: 7,       # 8–30 days
    2: 30,      # 31–90 days
    3: 90,      # 91–365 days
    4: 365      # >365 days
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_domain(value):
    """
    Extract and normalize a domain from:
      user@example.com
      Example <user@example.com>
      example.com
    """
    if not value:
        return None

    value = value.strip().lower()

    # Extract email address
    match = re.search(r'@([a-z0-9.-]+\.[a-z]{2,})', value)
    if match:
        return match.group(1).strip(".")

    # Otherwise assume it is already a domain
    value = re.sub(r'^https?://', '', value)
    value = value.split('/')[0]
    value = value.split(':')[0]

    return value.strip(".")


def extract_email_domain(header_value):
    if not header_value:
        return None

    match = re.search(
        r'[\w.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})',
        header_value
    )

    if match:
        return match.group(1).lower()

    return None


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(text):
    """
    Shannon entropy.
    """

    if not text:
        return 0.0

    frequency = {}

    for char in text:
        frequency[char] = frequency.get(char, 0) + 1

    length = len(text)

    entropy = 0.0

    for count in frequency.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


# ============================================================
# SPF
# ============================================================

def extract_spf(msg):
    """
    Extract SPF result from:
      Authentication-Results
      Received-SPF
    """

    auth_results = msg.get_all("Authentication-Results", [])

    for header in auth_results:
        match = re.search(
            r'\bspf\s*=\s*(pass|fail|softfail|neutral|none|temperror|permerror)',
            header,
            re.I
        )

        if match:
            return match.group(1).lower()

    received_spf = msg.get_all("Received-SPF", [])

    for header in received_spf:
        match = re.search(
            r'^\s*(pass|fail|softfail|neutral|none|temperror|permerror)\b',
            header,
            re.I
        )

        if match:
            return match.group(1).lower()

    return "unknown"


def spf_binary(result):
    return 1 if result == "pass" else 0


# ============================================================
# DKIM
# ============================================================

def extract_dkim(msg):
    """
    Extract:
      DKIM pass/fail
      DKIM alignment

    Alignment is inferred from:
      Authentication-Results
      dkim=pass ... header.d=domain
    """

    auth_results = msg.get_all("Authentication-Results", [])

    dkim_result = "unknown"
    dkim_domain = None

    for header in auth_results:

        match = re.search(
            r'\bdkim\s*=\s*(pass|fail|neutral|none|temperror|permerror)',
            header,
            re.I
        )

        if match:
            dkim_result = match.group(1).lower()

            # Extract header.d
            domain_match = re.search(
                r'\bheader\.d\s*=\s*([A-Za-z0-9.-]+)',
                header,
                re.I
            )

            if domain_match:
                dkim_domain = domain_match.group(1).lower()

            break

    # Fallback: read DKIM-Signature directly
    if not dkim_domain:

        dkim_headers = msg.get_all("DKIM-Signature", [])

        for header in dkim_headers:

            domain_match = re.search(
                r'\bd=([A-Za-z0-9.-]+)',
                header,
                re.I
            )

            if domain_match:
                dkim_domain = domain_match.group(1).lower()
                break

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    from_domain = extract_email_domain(msg.get("From", ""))

    if dkim_result == "pass" and dkim_domain and from_domain:

        # Exact organizational-domain comparison
        dkim_aligned = organizational_domain(dkim_domain) == \
                       organizational_domain(from_domain)

        dkim_aligned_value = 1 if dkim_aligned else 0

    else:
        dkim_aligned_value = 0

    return dkim_result, dkim_domain, dkim_aligned_value


# ============================================================
# DMARC
# ============================================================

def extract_dmarc(msg):
    auth_results = msg.get_all("Authentication-Results", [])

    for header in auth_results:

        match = re.search(
            r'\bdmarc\s*=\s*(pass|fail|neutral|none|temperror|permerror)',
            header,
            re.I
        )

        if match:
            return match.group(1).lower()

    return "unknown"


def dmarc_binary(result):
    return 1 if result == "pass" else 0


# ============================================================
# ORGANIZATIONAL DOMAIN
# ============================================================

def organizational_domain(domain):
    """
    Simplified organizational-domain extraction.

    Example:
        mail.google.com -> google.com
        login.microsoft.com -> microsoft.com

    For production use, replace this with publicsuffix2/tldextract
    to correctly handle domains such as co.uk.
    """

    if not domain:
        return None

    domain = domain.lower().strip(".")

    parts = domain.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


# ============================================================
# FROM / REPLY-TO MISMATCH
# ============================================================

def from_reply_to_mismatch(msg):

    from_domain = extract_email_domain(
        msg.get("From", "")
    )

    reply_domain = extract_email_domain(
        msg.get("Reply-To", "")
    )

    # No Reply-To = no mismatch
    if not reply_domain:
        return 0

    if not from_domain:
        return 0

    return int(
        organizational_domain(from_domain)
        != organizational_domain(reply_domain)
    )


# ============================================================
# FROM / RETURN-PATH MISMATCH
# ============================================================

def from_return_path_mismatch(msg):

    from_domain = extract_email_domain(
        msg.get("From", "")
    )

    return_path_domain = extract_email_domain(
        msg.get("Return-Path", "")
    )

    if not return_path_domain:
        return 0

    if not from_domain:
        return 0

    return int(
        organizational_domain(from_domain)
        != organizational_domain(return_path_domain)
    )


# ============================================================
# RECEIVED CHAIN
# ============================================================

def count_received_hops(msg):
    """
    Number of Received headers.
    """

    received = msg.get_all("Received", [])

    return len(received)


# ============================================================
# RECEIVED CHAIN UNUSUAL PATTERN
# ============================================================

def analyze_received_chain(msg):

    received = msg.get_all("Received", [])

    hops = len(received)

    if hops == 0:
        return {
            "received_hops": 0,
            "received_chain_label": 0,
            "unusual_received_chain": 0
        }

    unusual = 0

    # --------------------------------------------------------
    # Check IPs
    # --------------------------------------------------------

    ips = []

    for header in received:

        found_ips = re.findall(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            header
        )

        for ip in found_ips:

            try:
                ipaddress.ip_address(ip)
                ips.append(ip)
            except ValueError:
                pass

    # Too many different IPs can indicate unusual routing
    if len(set(ips)) >= 6:
        unusual = 1

    # --------------------------------------------------------
    # Check private IPs appearing in public-looking chain
    # --------------------------------------------------------

    for ip in ips:

        try:
            ip_obj = ipaddress.ip_address(ip)

            if ip_obj.is_private:
                unusual = 1

        except ValueError:
            pass

    # --------------------------------------------------------
    # Check duplicate/repeated Received entries
    # --------------------------------------------------------

    normalized = [
        re.sub(r'\s+', ' ', h.lower()).strip()
        for h in received
    ]

    if len(normalized) != len(set(normalized)):
        unusual = 1

    return {
        "received_hops": hops,
        "unusual_received_chain": unusual
    }


# ============================================================
# DOMAIN AGE
# ============================================================

def get_domain_creation_date(domain):

    if not domain:
        return None

    try:

        response = requests.get(
            f"https://rdap.org/domain/{domain}",
            timeout=5,
            headers={
                "User-Agent": "STS-feature-extractor/1.0"
            }
        )

        if response.status_code != 200:
            return None

        data = response.json()

        events = data.get("events", [])

        for event in events:

            if event.get("eventAction") == "registration":

                date_string = event.get("eventDate")

                if date_string:

                    return datetime.fromisoformat(
                        date_string.replace("Z", "+00:00")
                    )

    except Exception:
        pass

    return None


def domain_age_days(domain):

    creation_date = get_domain_creation_date(domain)

    if not creation_date:
        return None

    now = datetime.now(timezone.utc)

    return max(
        0,
        (now - creation_date).days
    )


def new_sender_domain_label(age_days):

    if age_days is None:
        return -1

    if age_days <= 7:
        return 0

    elif age_days <= 30:
        return 1

    elif age_days <= 90:
        return 2

    elif age_days <= 365:
        return 3

    else:
        return 4


# ============================================================
# MAIN STS EXTRACTION
# ============================================================

def extract_sts_features(eml_path):

    with open(
        eml_path,
        "rb"
    ) as f:

        msg = BytesParser(
            policy=policy.default
        ).parse(f)

    # --------------------------------------------------------
    # SPF
    # --------------------------------------------------------

    spf_result = extract_spf(msg)

    # --------------------------------------------------------
    # DKIM
    # --------------------------------------------------------

    dkim_result, dkim_domain, dkim_aligned = extract_dkim(msg)

    # --------------------------------------------------------
    # DMARC
    # --------------------------------------------------------

    dmarc_result = extract_dmarc(msg)

    # --------------------------------------------------------
    # From domain
    # --------------------------------------------------------

    from_domain = extract_email_domain(
        msg.get("From", "")
    )

    # --------------------------------------------------------
    # Domain entropy
    # --------------------------------------------------------

    sender_domain_entropy = calculate_entropy(
        from_domain
        if from_domain
        else ""
    )

    # --------------------------------------------------------
    # Sender domain age
    # --------------------------------------------------------

    age_days = domain_age_days(
        from_domain
    )

    new_domain = new_sender_domain_label(
        age_days
    )

    # --------------------------------------------------------
    # Received chain
    # --------------------------------------------------------

    received_info = analyze_received_chain(
        msg
    )

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = {

        # SPF
        "spf_pass": spf_binary(spf_result),

        # DKIM
        "dkim_pass": int(dkim_result == "pass"),
        "dkim_aligned": dkim_aligned,

        # DMARC
        "dmarc_pass": dmarc_binary(dmarc_result),

        # Header mismatches
        "from_reply_to_mismatch":
            from_reply_to_mismatch(msg),

        "from_return_path_mismatch":
            from_return_path_mismatch(msg),

        # Sender domain
        "new_sender_domain":
            new_domain,

        # Entropy
        "sender_domain_entropy":
            sender_domain_entropy,

        # Received chain
        "received_hops":
            received_info["received_hops"],

        "unusual_received_chain":
            received_info["unusual_received_chain"],
    }

    # --------------------------------------------------------
    # Useful metadata
    # --------------------------------------------------------

    metadata = {

        "from_domain":
            from_domain,

        "dkim_domain":
            dkim_domain,

        "spf_result":
            spf_result,

        "dkim_result":
            dkim_result,

        "dmarc_result":
            dmarc_result,

        "domain_age_days":
            age_days
    }

    return features, metadata


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    eml_file = "sample.eml"

    sts_features, sts_metadata = extract_sts_features(
        eml_file
    )

    print("\n========== STS FEATURES ==========\n")

    for key, value in sts_features.items():
        print(f"{key}: {value}")

    print("\n========== METADATA ==========\n")

    for key, value in sts_metadata.items():
        print(f"{key}: {value}")