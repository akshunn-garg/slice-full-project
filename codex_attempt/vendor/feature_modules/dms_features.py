import re
import email
import requests
import ipaddress

from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse, unquote


# ============================================================
# 1. DOMAIN HELPERS
# ============================================================

def organizational_domain(domain):
    """Get simplified registered/organizational domain."""
    if not domain:
        return None

    domain = domain.lower().strip(".")
    parts = domain.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


def registered_domain_from_url(url):
    """Extract simplified registered domain from URL."""
    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return None

        hostname = hostname.lower()

        # IP address
        try:
            ipaddress.ip_address(hostname)
            return hostname
        except ValueError:
            pass

        return organizational_domain(hostname)

    except Exception:
        return None


def hostname_from_url(url):
    try:
        return urlparse(url).hostname
    except Exception:
        return None


# ============================================================
# 2. EMAIL INFORMATION
# ============================================================

def get_from_domain(msg):
    from_header = msg.get("From", "")

    match = re.search(r'@([A-Za-z0-9.-]+)', from_header)

    if match:
        return match.group(1).lower()

    return None


def get_dkim_domain(msg):
    """Extract DKIM signing domain."""

    # First try Authentication-Results
    auth_results = msg.get_all("Authentication-Results", [])

    for header in auth_results:
        match = re.search(
            r'dkim\s*=\s*(?:pass|fail|softfail|neutral|none|temperror|permerror)'
            r'.*?\bd=([A-Za-z0-9.-]+)',
            header,
            re.I
        )

        if match:
            return match.group(1).lower()

    # Fallback to DKIM-Signature
    dkim_signature = msg.get("DKIM-Signature", "")

    match = re.search(
        r'\bd=([A-Za-z0-9.-]+)',
        dkim_signature,
        re.I
    )

    if match:
        return match.group(1).lower()

    return None


# ============================================================
# 3. HTML EXTRACTION
# ============================================================

def get_html(msg):

    if msg.is_multipart():

        for part in msg.walk():

            content_type = part.get_content_type()

            if content_type == "text/html":

                try:
                    return part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)

                    if payload:
                        return payload.decode(
                            part.get_content_charset() or "utf-8",
                            errors="ignore"
                        )

    else:

        if msg.get_content_type() == "text/html":

            try:
                return msg.get_content()
            except Exception:
                pass

    return ""


def extract_links(html):

    soup = BeautifulSoup(html, "html.parser")

    links = []

    for tag in soup.find_all("a"):

        href = tag.get("href")

        if not href:
            continue

        href = href.strip()

        if not href.lower().startswith(("http://", "https://")):
            continue

        anchor_text = tag.get_text(" ", strip=True)

        links.append({
            "url": href,
            "anchor": anchor_text,
            "tag": tag
        })

    return links


# ============================================================
# 4. VISIBLE URL
# ============================================================

def normalize_url(url):

    if not url:
        return ""

    url = unquote(url).strip().lower()

    # Remove trailing slash
    url = url.rstrip("/")

    return url


def extract_visible_url(anchor_text):

    if not anchor_text:
        return None

    text = anchor_text.strip()

    # Look for a URL inside visible text
    match = re.search(
        r'https?://[^\s<>"\']+',
        text,
        re.I
    )

    if match:
        return match.group(0).rstrip(".,);]")

    return None


def visible_url_equals_actual_target(anchor, actual_url):

    visible_url = extract_visible_url(anchor)

    if not visible_url:
        return 0

    return int(
        normalize_url(visible_url)
        == normalize_url(actual_url)
    )


# ============================================================
# 5. ANCHOR == DESTINATION
# ============================================================

def anchor_equals_destination(anchor, destination_domain):

    if not anchor or not destination_domain:
        return 0

    anchor = anchor.lower().strip()

    destination_domain = destination_domain.lower()

    # If anchor itself contains a URL
    visible_url = extract_visible_url(anchor)

    if visible_url:

        visible_domain = registered_domain_from_url(
            visible_url
        )

        return int(
            visible_domain == destination_domain
        )

    # Otherwise compare text with domain name
    destination_name = destination_domain.split(".")[0]

    normalized_anchor = re.sub(
        r'[^a-z0-9]',
        '',
        anchor
    )

    normalized_destination = re.sub(
        r'[^a-z0-9]',
        '',
        destination_name
    )

    return int(
        normalized_anchor == normalized_destination
    )


# ============================================================
# 6. BUTTON PURPOSE == DESTINATION
# ============================================================

def infer_button_purpose(anchor_text):

    if not anchor_text:
        return None

    text = anchor_text.lower().strip()

    # Common purposes
    if any(x in text for x in [
        "login",
        "log in",
        "sign in",
        "signin"
    ]):
        return "login"

    if any(x in text for x in [
        "verify",
        "verification",
        "confirm"
    ]):
        return "verification"

    if any(x in text for x in [
        "payment",
        "pay",
        "billing",
        "invoice"
    ]):
        return "payment"

    if any(x in text for x in [
        "refund"
    ]):
        return "refund"

    if any(x in text for x in [
        "order",
        "orders",
        "purchase"
    ]):
        return "order"

    if any(x in text for x in [
        "account",
        "profile"
    ]):
        return "account"

    if any(x in text for x in [
        "unsubscribe"
    ]):
        return "unsubscribe"

    if any(x in text for x in [
        "update"
    ]):
        return "update"

    return None


def infer_destination_purpose(destination_domain, url):

    if not destination_domain:
        return None

    url_lower = url.lower()

    # Infer purpose from URL path/query
    purpose_keywords = {

        "login": [
            "login",
            "signin",
            "sign-in"
        ],

        "verification": [
            "verify",
            "verification",
            "confirm"
        ],

        "payment": [
            "payment",
            "pay",
            "billing",
            "invoice"
        ],

        "refund": [
            "refund"
        ],

        "order": [
            "order",
            "orders",
            "purchase"
        ],

        "account": [
            "account",
            "profile"
        ],

        "unsubscribe": [
            "unsubscribe"
        ],

        "update": [
            "update"
        ]
    }

    for purpose, keywords in purpose_keywords.items():

        for keyword in keywords:

            if keyword in url_lower:
                return purpose

    return None


def button_purpose_equals_destination(anchor, destination_domain, url):

    button_purpose = infer_button_purpose(anchor)

    if button_purpose is None:
        return 0

    destination_purpose = infer_destination_purpose(
        destination_domain,
        url
    )

    if destination_purpose is None:
        return 0

    return int(
        button_purpose == destination_purpose
    )


# ============================================================
# 7. BRAND IN SUBDOMAIN BUT NOT REGISTERED DOMAIN
# ============================================================

def normalize_brand(brand):

    if not brand:
        return ""

    return re.sub(
        r'[^a-z0-9]',
        '',
        brand.lower()
    )


def brand_in_subdomain_but_not_registered(
    claimed_brand,
    url
):

    if not claimed_brand:
        return 0

    hostname = hostname_from_url(url)

    if not hostname:
        return 0

    hostname = hostname.lower()

    parts = hostname.split(".")

    if len(parts) < 3:
        return 0

    registered = organizational_domain(hostname)

    if not registered:
        return 0

    brand = normalize_brand(claimed_brand)

    if not brand:
        return 0

    # Brand appears somewhere before registered domain
    subdomain_parts = parts[:-2]

    brand_in_subdomain = any(
        brand in normalize_brand(part)
        for part in subdomain_parts
    )

    # Brand should NOT correspond to registered domain
    registered_name = registered.split(".")[0]

    brand_matches_registered = (
        brand == normalize_brand(registered_name)
    )

    return int(
        brand_in_subdomain
        and not brand_matches_registered
    )


# ============================================================
# 8. DKIM DOMAIN == DESTINATION
# ============================================================

def dkim_domain_equals_destination(
    dkim_domain,
    destination_domain
):

    if not dkim_domain or not destination_domain:
        return 0

    return int(
        organizational_domain(dkim_domain)
        ==
        organizational_domain(destination_domain)
    )


# ============================================================
# 9. KNOWN URL SHORTENERS
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


def is_known_shortener(url):

    domain = registered_domain_from_url(url)

    if not domain:
        return 0

    return int(
        domain.lower() in KNOWN_SHORTENERS
    )


# ============================================================
# 10. FOLLOW REDIRECTS
# ============================================================

def get_final_redirect_domain(url):

    try:

        response = requests.get(
            url,
            allow_redirects=True,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        final_url = response.url

        return registered_domain_from_url(
            final_url
        )

    except Exception:

        # If redirect cannot be followed,
        # use original destination
        return registered_domain_from_url(url)


# ============================================================
# 11. EXPECTED DOMAIN
# ============================================================

def get_expected_domain(
    claimed_brand,
    from_domain,
    dkim_domain
):

    # Prefer DKIM authenticated domain
    if dkim_domain:
        return organizational_domain(
            dkim_domain
        )

    if from_domain:
        return organizational_domain(
            from_domain
        )

    return None


# ============================================================
# 12. FINAL DOMAIN == EXPECTED DOMAIN
# ============================================================

def final_domain_equals_expected(
    final_domain,
    expected_domain
):

    if not final_domain or not expected_domain:
        return 0

    return int(
        organizational_domain(final_domain)
        ==
        organizational_domain(expected_domain)
    )


# ============================================================
# 13. CLAIMED BRAND
# ============================================================

def extract_claimed_brand(msg):

    # Same approach as current BAS implementation:
    # use visible sender name as claimed brand.

    from_header = msg.get("From", "")

    match = re.match(
        r'^\s*"?([^"<]+)"?\s*<',
        from_header
    )

    if match:
        return match.group(1).strip()

    # Handle From: Google <email@example.com>
    if "@" in from_header:

        name_part = from_header.split("<")[0].strip()

        name_part = name_part.strip('"')

        if name_part:
            return name_part

    return None


# ============================================================
# 14. MAIN DMS EXTRACTION
# ============================================================

def extract_dms_features(eml_path):

    # --------------------------------------------------------
    # Read EML
    # --------------------------------------------------------

    with open(eml_path, "rb") as f:

        msg = BytesParser(
            policy=policy.default
        ).parse(f)

    # --------------------------------------------------------
    # Email-level information
    # --------------------------------------------------------

    from_domain = get_from_domain(msg)

    dkim_domain = get_dkim_domain(msg)

    claimed_brand = extract_claimed_brand(msg)

    expected_domain = get_expected_domain(
        claimed_brand,
        from_domain,
        dkim_domain
    )

    # --------------------------------------------------------
    # Extract HTML links
    # --------------------------------------------------------

    html = get_html(msg)

    links = extract_links(html)

    link_results = []

    # --------------------------------------------------------
    # Analyze every link
    # --------------------------------------------------------

    for i, link in enumerate(links, start=1):

        url = link["url"]

        anchor = link["anchor"]

        # Actual destination
        destination_domain = registered_domain_from_url(
            url
        )

        # Final redirect destination
        final_domain = get_final_redirect_domain(
            url
        )

        # -----------------------------------------------
        # DMS FEATURES FOR THIS LINK
        # -----------------------------------------------

        f1 = visible_url_equals_actual_target(
            anchor,
            url
        )

        f2 = anchor_equals_destination(
            anchor,
            destination_domain
        )

        f3 = button_purpose_equals_destination(
            anchor,
            destination_domain,
            url
        )

        f4 = brand_in_subdomain_but_not_registered(
            claimed_brand,
            url
        )

        f5 = dkim_domain_equals_destination(
            dkim_domain,
            destination_domain
        )

        f6 = is_known_shortener(
            url
        )

        f7 = final_domain_equals_expected(
            final_domain,
            expected_domain
        )

        link_results.append({

            "link_number": i,

            "url": url,

            "anchor": anchor,

            "destination_domain": destination_domain,

            "final_redirect_domain": final_domain,

            "visible_url_equals_actual_target": f1,

            "anchor_equals_destination": f2,

            "button_purpose_equals_destination": f3,

            "brand_in_subdomain_but_not_registered": f4,

            "dkim_domain_equals_destination": f5,

            "url_is_known_shortener": f6,

            "final_domain_equals_expected": f7
        })

    # ========================================================
    # AGGREGATE LINK FEATURES
    # ========================================================

    n = len(link_results)

    if n == 0:

        feature_1 = 0.0
        feature_2 = 0.0
        feature_3 = 0.0
        feature_4 = 0.0
        feature_5 = 0.0
        feature_6 = 0.0
        feature_7 = 0.0

    else:

        feature_1 = sum(
            x["visible_url_equals_actual_target"]
            for x in link_results
        ) / n

        feature_2 = sum(
            x["anchor_equals_destination"]
            for x in link_results
        ) / n

        feature_3 = sum(
            x["button_purpose_equals_destination"]
            for x in link_results
        ) / n

        feature_4 = sum(
            x["brand_in_subdomain_but_not_registered"]
            for x in link_results
        ) / n

        feature_5 = sum(
            x["dkim_domain_equals_destination"]
            for x in link_results
        ) / n

        feature_6 = sum(
            x["url_is_known_shortener"]
            for x in link_results
        ) / n

        feature_7 = sum(
            x["final_domain_equals_expected"]
            for x in link_results
        ) / n

    # ========================================================
    # FINAL DMS VECTOR
    # ========================================================

    features = {

        "visible_url_equals_actual_target": feature_1,

        "anchor_equals_destination": feature_2,

        "button_purpose_equals_destination": feature_3,

        "brand_in_subdomain_but_not_registered": feature_4,

        "dkim_domain_equals_destination": feature_5,

        "url_is_known_shortener": feature_6,

        "final_domain_equals_expected": feature_7
    }

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "claimed_brand": claimed_brand,

        "from_domain": from_domain,

        "dkim_domain": dkim_domain,

        "expected_domain": expected_domain,

        "number_of_links": n,

        "link_results": link_results
    }

    return features, metadata


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    eml_path = "sample.eml"

    features, metadata = extract_dms_features(
        eml_path
    )

    print("\n========== DMS FEATURES ==========")

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )

    print("\n========== DMS METADATA ==========")

    print(
        "Claimed brand:",
        metadata["claimed_brand"]
    )

    print(
        "From domain:",
        metadata["from_domain"]
    )

    print(
        "DKIM domain:",
        metadata["dkim_domain"]
    )

    print(
        "Expected domain:",
        metadata["expected_domain"]
    )

    print(
        "Number of links:",
        metadata["number_of_links"]
    )

    print("\n========== INDIVIDUAL LINKS ==========")

    for link in metadata["link_results"]:

        print(
            f"\nLink {link['link_number']}"
        )

        print(
            "URL:",
            link["url"]
        )

        print(
            "Anchor:",
            link["anchor"]
        )

        print(
            "Destination:",
            link["destination_domain"]
        )

        print(
            "Final redirect:",
            link["final_redirect_domain"]
        )