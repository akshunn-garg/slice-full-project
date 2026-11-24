import re
import math
import ipaddress

from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup


# ============================================================
# 1. KNOWN URL SHORTENERS
# ============================================================

KNOWN_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "cutt.ly",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "shorturl.at",
    "tiny.cc",
    "lnkd.in",
    "rb.gy",
    "s.id",
    "soo.gd",
    "bl.ink",
    "clck.ru",
    "amzn.to",
    "youtu.be",
    "c.gle"
}


# ============================================================
# 2. URL / DOMAIN HELPERS
# ============================================================

def get_hostname(url):
    try:
        return urlparse(url).hostname
    except Exception:
        return None


def get_registered_domain(hostname):
    if not hostname:
        return None

    hostname = hostname.lower().strip(".")

    # IP address
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass

    parts = hostname.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname


def get_subdomain_count(url):
    hostname = get_hostname(url)

    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 0
    except ValueError:
        pass

    registered = get_registered_domain(hostname)

    if not registered:
        return 0

    hostname_parts = hostname.split(".")
    registered_parts = registered.split(".")

    return max(
        0,
        len(hostname_parts) - len(registered_parts)
    )


# ============================================================
# 3. URL LENGTH — 5 LABELS
# ============================================================

def url_length_label(url):

    length = len(url)

    # 0 = very short
    # 1 = short
    # 2 = medium
    # 3 = long
    # 4 = extremely long

    if length <= 50:
        return 0

    elif length <= 100:
        return 1

    elif length <= 200:
        return 2

    elif length <= 300:
        return 3

    else:
        return 4


# ============================================================
# 4. STRING ENTROPY — CONTINUOUS
# ============================================================

def string_entropy(url):

    if not url:
        return 0.0

    counts = {}

    for char in url:
        counts[char] = counts.get(char, 0) + 1

    length = len(url)

    entropy = 0.0

    for count in counts.values():

        probability = count / length

        entropy -= (
            probability *
            math.log2(probability)
        )

    return entropy


# ============================================================
# 5. ENCODED CHARACTERS — BINARY
# ============================================================

def has_encoded_characters(url):

    # Percent encoding such as:
    # %20
    # %2F
    # %3D
    # %40

    if re.search(
        r'%[0-9A-Fa-f]{2}',
        url
    ):
        return 1

    # HTML entities such as:
    # &#x2F;
    # &#47;
    # &amp;

    if re.search(
        r'&(?:#x[0-9A-Fa-f]+|#\d+|\w+);',
        url
    ):
        return 1

    return 0


# ============================================================
# 6. IP ADDRESS — BINARY
# ============================================================

def is_ip_address(url):

    hostname = get_hostname(url)

    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1

    except ValueError:
        return 0


# ============================================================
# 7. PUNYCODE — BINARY
# ============================================================

def has_punycode(url):

    hostname = get_hostname(url)

    if not hostname:
        return 0

    # xn-- is the standard punycode prefix
    return int(
        any(
            part.lower().startswith("xn--")
            for part in hostname.split(".")
        )
    )


# ============================================================
# 8. NON-STANDARD PORT — BINARY
# ============================================================

def has_nonstandard_port(url):

    try:

        parsed = urlparse(url)

        port = parsed.port

        if port is None:
            return 0

        # Standard ports
        if parsed.scheme.lower() == "http" and port == 80:
            return 0

        if parsed.scheme.lower() == "https" and port == 443:
            return 0

        return 1

    except Exception:

        return 0


# ============================================================
# 9. REDIRECT-LIKE PARAMETERS — BINARY
# ============================================================

REDIRECT_PARAMETERS = {

    "url",
    "uri",
    "redirect",
    "redirect_url",
    "redirect_uri",
    "redir",
    "target",
    "destination",
    "dest",
    "next",
    "next_url",
    "return",
    "return_url",
    "returnto",
    "return_to",
    "continue",
    "continue_url",
    "goto",
    "go",
    "link",
    "out",
    "forward",
    "forward_url"
}


def has_redirect_like_parameters(url):

    try:

        parsed = urlparse(url)

        query = parsed.query.lower()

        if not query:
            return 0

        parameters = query.split("&")

        for parameter in parameters:

            key = parameter.split("=")[0]

            key = unquote(key).lower().strip()

            if key in REDIRECT_PARAMETERS:
                return 1

        return 0

    except Exception:

        return 0


# ============================================================
# 10. KNOWN SHORTENER — BINARY
# ============================================================

def is_known_shortener(url):

    hostname = get_hostname(url)

    if not hostname:
        return 0

    registered = get_registered_domain(hostname)

    if not registered:
        return 0

    return int(
        registered.lower() in KNOWN_SHORTENERS
    )


# ============================================================
# 11. MISLEADING BRAND TOKEN — BINARY
# ============================================================

def normalize_token(text):

    if not text:
        return ""

    return re.sub(
        r'[^a-z0-9]',
        '',
        text.lower()
    )


def misleading_brand_token(
    url,
    claimed_brand,
    expected_domain=None
):

    if not claimed_brand:
        return 0

    brand = normalize_token(
        claimed_brand
    )

    if not brand:
        return 0

    hostname = get_hostname(url)

    if not hostname:
        return 0

    registered_domain = get_registered_domain(
        hostname
    )

    if not registered_domain:
        return 0

    registered_name = registered_domain.split(".")[0]

    # If the actual registered domain belongs to the
    # claimed brand, this is NOT misleading.
    if brand == normalize_token(
        registered_name
    ):
        return 0

    # Check hostname and URL for brand token
    url_without_scheme = re.sub(
        r'^https?://',
        '',
        url.lower()
    )

    # Look for brand in hostname/path/query
    if brand in normalize_token(
        url_without_scheme
    ):

        # Make sure it isn't simply the legitimate
        # registered domain.
        if brand not in normalize_token(
            registered_name
        ):
            return 1

    return 0


# ============================================================
# 12. EXTRACT CLAIMED BRAND
# ============================================================

def extract_claimed_brand(msg):

    from_header = msg.get(
        "From",
        ""
    )

    # Example:
    # Google Play <something@google.com>

    match = re.match(
        r'^\s*"?([^"<]+)"?\s*<',
        from_header
    )

    if match:

        return match.group(1).strip()

    # Fallback
    if "@" in from_header:

        name_part = (
            from_header
            .split("<")[0]
            .strip()
            .strip('"')
        )

        if name_part:
            return name_part

    return None


# ============================================================
# 13. EXTRACT HTML
# ============================================================

def get_html(msg):

    if msg.is_multipart():

        for part in msg.walk():

            if part.get_content_type() == "text/html":

                try:
                    return part.get_content()

                except Exception:

                    payload = part.get_payload(
                        decode=True
                    )

                    if payload:

                        return payload.decode(
                            part.get_content_charset()
                            or "utf-8",
                            errors="ignore"
                        )

    else:

        if msg.get_content_type() == "text/html":

            try:
                return msg.get_content()

            except Exception:
                pass

    return ""


# ============================================================
# 14. EXTRACT LINKS
# ============================================================

def extract_links(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    links = []

    for tag in soup.find_all("a"):

        href = tag.get("href")

        if not href:
            continue

        href = href.strip()

        if not href.lower().startswith(
            ("http://", "https://")
        ):
            continue

        links.append(href)

    return links


# ============================================================
# 15. LOS FOR ONE URL
# ============================================================

def analyze_url(
    url,
    claimed_brand,
    expected_domain=None
):

    return {

        "url_length_label":
            url_length_label(url),

        "string_entropy":
            string_entropy(url),

        "encoded_characters":
            has_encoded_characters(url),

        "ip_address":
            is_ip_address(url),

        "punycode":
            has_punycode(url),

        "subdomain_count":
            get_subdomain_count(url),

        "nonstandard_port":
            has_nonstandard_port(url),

        "redirect_like_parameters":
            has_redirect_like_parameters(url),

        "known_shortener":
            is_known_shortener(url),

        "misleading_brand_token":
            misleading_brand_token(
                url,
                claimed_brand,
                expected_domain
            )
    }


# ============================================================
# 16. MAIN LOS EXTRACTION
# ============================================================

def extract_los_features(eml_path):

    # --------------------------------------------------------
    # Read EML
    # --------------------------------------------------------

    with open(
        eml_path,
        "rb"
    ) as f:

        msg = BytesParser(
            policy=policy.default
        ).parse(f)

    # --------------------------------------------------------
    # Email information
    # --------------------------------------------------------

    claimed_brand = extract_claimed_brand(
        msg
    )

    # Expected domain from From header
    from_header = msg.get(
        "From",
        ""
    )

    match = re.search(
        r'@([A-Za-z0-9.-]+)',
        from_header
    )

    from_domain = (
        match.group(1).lower()
        if match
        else None
    )

    expected_domain = (
        get_registered_domain(from_domain)
        if from_domain
        else None
    )

    # --------------------------------------------------------
    # Extract links
    # --------------------------------------------------------

    html = get_html(msg)

    links = extract_links(html)

    # --------------------------------------------------------
    # Analyze each URL
    # --------------------------------------------------------

    link_results = []

    for i, url in enumerate(
        links,
        start=1
    ):

        features = analyze_url(
            url,
            claimed_brand,
            expected_domain
        )

        link_results.append({

            "link_number": i,

            "url": url,

            **features

        })

    # ========================================================
    # AGGREGATE
    # ========================================================

    n = len(link_results)

    if n == 0:

        features = {

            "url_length_label": 0.0,

            "string_entropy": 0.0,

            "encoded_characters": 0.0,

            "ip_address": 0.0,

            "punycode": 0.0,

            "subdomain_count": 0.0,

            "nonstandard_port": 0.0,

            "redirect_like_parameters": 0.0,

            "known_shortener": 0.0,

            "misleading_brand_token": 0.0
        }

    else:

        # For categorical/binary features:
        # proportion of links satisfying condition.

        features = {

            "url_length_label":
                sum(
                    x["url_length_label"]
                    for x in link_results
                ) / n,

            "string_entropy":
                sum(
                    x["string_entropy"]
                    for x in link_results
                ) / n,

            "encoded_characters":
                sum(
                    x["encoded_characters"]
                    for x in link_results
                ) / n,

            "ip_address":
                sum(
                    x["ip_address"]
                    for x in link_results
                ) / n,

            "punycode":
                sum(
                    x["punycode"]
                    for x in link_results
                ) / n,

            "subdomain_count":
                sum(
                    x["subdomain_count"]
                    for x in link_results
                ) / n,

            "nonstandard_port":
                sum(
                    x["nonstandard_port"]
                    for x in link_results
                ) / n,

            "redirect_like_parameters":
                sum(
                    x["redirect_like_parameters"]
                    for x in link_results
                ) / n,

            "known_shortener":
                sum(
                    x["known_shortener"]
                    for x in link_results
                ) / n,

            "misleading_brand_token":
                sum(
                    x["misleading_brand_token"]
                    for x in link_results
                ) / n
        }

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "claimed_brand":
            claimed_brand,

        "expected_domain":
            expected_domain,

        "number_of_links":
            n,

        "link_results":
            link_results
    }

    return features, metadata


# ============================================================
# 17. RUN
# ============================================================

if __name__ == "__main__":

    eml_path = "sample.eml"

    features, metadata = extract_los_features(
        eml_path
    )

    print(
        "\n========== LOS FEATURES =========="
    )

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )

    print(
        "\n========== LOS METADATA =========="
    )

    print(
        "Claimed brand:",
        metadata["claimed_brand"]
    )

    print(
        "Expected domain:",
        metadata["expected_domain"]
    )

    print(
        "Number of links:",
        metadata["number_of_links"]
    )

    print(
        "\n========== INDIVIDUAL LINKS =========="
    )

    for link in metadata["link_results"]:

        print(
            f"\nLink {link['link_number']}"
        )

        for key, value in link.items():

            if key != "link_number":

                print(
                    f"{key}: {value}"
                )