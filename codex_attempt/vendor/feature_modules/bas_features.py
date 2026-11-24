import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from email import policy
from email.parser import BytesParser


# ============================================================
# BASIC HELPERS
# ============================================================

def extract_email_domain(value):
    if not value:
        return None

    match = re.search(
        r'[\w.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})',
        value
    )

    if match:
        return match.group(1).lower()

    return None


def organizational_domain(domain):
    """
    Simple registered/organizational domain extraction.

    amazon.com              -> amazon.com
    login.amazon.com        -> amazon.com
    """

    if not domain:
        return None

    domain = domain.lower().strip(".")

    parts = domain.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


def registered_domain_from_url(url):

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return None

        return organizational_domain(hostname)

    except Exception:
        return None


def normalize_brand(text):

    if not text:
        return ""

    text = text.lower()

    # Remove spaces, punctuation and symbols
    return re.sub(r'[^a-z0-9]', '', text)


# ============================================================
# SENDER INFORMATION
# ============================================================

def extract_sender_name(msg):

    from_header = msg.get("From", "")

    match = re.match(
        r'^\s*(.*?)\s*<',
        from_header
    )

    if match:
        return match.group(1).strip()

    return ""


def get_from_domain(msg):

    return extract_email_domain(
        msg.get("From", "")
    )


# ============================================================
# DKIM DOMAIN
# ============================================================

def get_dkim_domain(msg):

    auth_results = msg.get_all(
        "Authentication-Results",
        []
    )

    for header in auth_results:

        # Make sure we are looking at the DKIM result
        dkim_match = re.search(
            r'\bdkim\s*=\s*(pass|fail|neutral|none|temperror|permerror)',
            header,
            re.I
        )

        if dkim_match:

            domain_match = re.search(
                r'\bheader\.d\s*=\s*([A-Za-z0-9.-]+)',
                header,
                re.I
            )

            if domain_match:
                return domain_match.group(1).lower()

    # Fallback to DKIM-Signature
    dkim_headers = msg.get_all(
        "DKIM-Signature",
        []
    )

    for header in dkim_headers:

        domain_match = re.search(
            r'\bd=([A-Za-z0-9.-]+)',
            header,
            re.I
        )

        if domain_match:
            return domain_match.group(1).lower()

    return None


# ============================================================
# CLAIMED BRAND
# ============================================================

def extract_claimed_brand(msg):

    """
    For now, claimed brand is taken from the
    visible sender/display name.

    Example:
        Amazon Security <security@amazon.com>
        -> Amazon Security
    """

    sender_name = extract_sender_name(msg)

    if sender_name:
        return sender_name

    return None


# ============================================================
# HTML EXTRACTION
# ============================================================

def get_html(msg):

    html_parts = []

    if msg.is_multipart():

        for part in msg.walk():

            if part.get_content_type() == "text/html":

                try:
                    html_parts.append(
                        part.get_content()
                    )

                except Exception:

                    payload = part.get_payload(
                        decode=True
                    )

                    if payload:
                        html_parts.append(
                            payload.decode(
                                errors="ignore"
                            )
                        )

    else:

        if msg.get_content_type() == "text/html":

            try:
                html_parts.append(
                    msg.get_content()
                )

            except Exception:
                pass

    return "\n".join(html_parts)


# ============================================================
# EXTRACT ALL LINKS
# ============================================================

def extract_links(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    links = []

    for a in soup.find_all("a"):

        href = a.get("href")

        if not href:
            continue

        href = href.strip()

        # Ignore non-web links
        if href.lower().startswith(
            (
                "mailto:",
                "javascript:",
                "#"
            )
        ):
            continue

        # Only HTTP/HTTPS
        if not href.lower().startswith(
            (
                "http://",
                "https://"
            )
        ):
            continue

        anchor_text = a.get_text(
            " ",
            strip=True
        )

        links.append({
            "anchor_text": anchor_text,
            "url": href
        })

    return links


# ============================================================
# FINAL REDIRECT DOMAIN
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

        return registered_domain_from_url(
            response.url
        )

    except Exception:

        # If request fails, use original destination
        return registered_domain_from_url(url)


# ============================================================
# FEATURE 1
# ============================================================

def sender_name_equals_claimed_brand(
    sender_name,
    claimed_brand
):

    if not sender_name or not claimed_brand:
        return 0

    return int(
        normalize_brand(sender_name)
        ==
        normalize_brand(claimed_brand)
    )


# ============================================================
# FEATURE 2
# ============================================================

def from_domain_equals_dkim_domain(
    from_domain,
    dkim_domain
):

    if not from_domain or not dkim_domain:
        return 0

    return int(
        organizational_domain(from_domain)
        ==
        organizational_domain(dkim_domain)
    )


# ============================================================
# LINK-LEVEL FEATURE 3
#
# claimed brand == anchor text
# ============================================================

def link_brand_equals_anchor(
    claimed_brand,
    anchor_text
):

    if not claimed_brand or not anchor_text:
        return 0

    return int(
        normalize_brand(claimed_brand)
        ==
        normalize_brand(anchor_text)
    )


# ============================================================
# LINK-LEVEL FEATURE 4
#
# claimed brand == destination registered domain
# ============================================================

def link_brand_equals_destination(
    claimed_brand,
    destination_domain
):

    if not claimed_brand or not destination_domain:
        return 0

    brand = normalize_brand(
        claimed_brand
    )

    destination_brand = normalize_brand(
        destination_domain.split(".")[0]
    )

    return int(
        brand == destination_brand
    )


# ============================================================
# LINK-LEVEL FEATURE 5
#
# authenticated domain == destination
# ============================================================

def link_authenticated_equals_destination(
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
# LINK-LEVEL FEATURE 6
#
# claimed/authenticated brand == final redirect
# ============================================================

def link_brand_equals_final_redirect(
    claimed_brand,
    dkim_domain,
    final_domain
):

    if not final_domain:
        return 0

    final_brand = normalize_brand(
        final_domain.split(".")[0]
    )

    # Check claimed brand
    if claimed_brand:

        claimed = normalize_brand(
            claimed_brand
        )

        if claimed == final_brand:
            return 1

    # Check authenticated/DKIM brand
    if dkim_domain:

        authenticated = normalize_brand(
            organizational_domain(
                dkim_domain
            ).split(".")[0]
        )

        if authenticated == final_brand:
            return 1

    return 0


# ============================================================
# MAIN BAS EXTRACTION
# ============================================================

def extract_bas_features(eml_path):

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
    # Email-level information
    # --------------------------------------------------------

    sender_name = extract_sender_name(msg)

    from_domain = get_from_domain(msg)

    dkim_domain = get_dkim_domain(msg)

    claimed_brand = extract_claimed_brand(msg)

    # --------------------------------------------------------
    # Extract all links
    # --------------------------------------------------------

    html = get_html(msg)

    links = extract_links(html)

    # --------------------------------------------------------
    # EMAIL-LEVEL BAS FEATURES
    # --------------------------------------------------------

    feature_1 = sender_name_equals_claimed_brand(
        sender_name,
        claimed_brand
    )

    feature_2 = from_domain_equals_dkim_domain(
        from_domain,
        dkim_domain
    )

    # --------------------------------------------------------
    # LINK-LEVEL FEATURES
    # --------------------------------------------------------

    link_results = []

    for link in links:

        anchor_text = link["anchor_text"]

        original_url = link["url"]

        # Destination before redirects
        destination_domain = (
            registered_domain_from_url(
                original_url
            )
        )

        # Final destination after redirects
        final_domain = (
            get_final_redirect_domain(
                original_url
            )
        )

        # Feature 3
        f3 = link_brand_equals_anchor(
            claimed_brand,
            anchor_text
        )

        # Feature 4
        f4 = link_brand_equals_destination(
            claimed_brand,
            destination_domain
        )

        # Feature 5
        f5 = link_authenticated_equals_destination(
            dkim_domain,
            destination_domain
        )

        # Feature 6
        f6 = link_brand_equals_final_redirect(
            claimed_brand,
            dkim_domain,
            final_domain
        )

        link_results.append({

            "anchor_text":
                anchor_text,

            "url":
                original_url,

            "destination_domain":
                destination_domain,

            "final_redirect_domain":
                final_domain,

            "brand_equals_anchor":
                f3,

            "brand_equals_destination":
                f4,

            "authenticated_equals_destination":
                f5,

            "brand_equals_final_redirect":
                f6
        })

    # --------------------------------------------------------
    # AGGREGATE LINK FEATURES
    # --------------------------------------------------------

    number_of_links = len(link_results)

    if number_of_links == 0:

        feature_3 = 0.0
        feature_4 = 0.0
        feature_5 = 0.0
        feature_6 = 0.0

    else:

        feature_3 = (
            sum(
                x["brand_equals_anchor"]
                for x in link_results
            )
            / number_of_links
        )

        feature_4 = (
            sum(
                x["brand_equals_destination"]
                for x in link_results
            )
            / number_of_links
        )

        feature_5 = (
            sum(
                x["authenticated_equals_destination"]
                for x in link_results
            )
            / number_of_links
        )

        feature_6 = (
            sum(
                x["brand_equals_final_redirect"]
                for x in link_results
            )
            / number_of_links
        )

    # --------------------------------------------------------
    # FINAL BAS VECTOR
    # --------------------------------------------------------

    features = {

        # Email-level
        "sender_name_equals_claimed_brand":
            feature_1,

        "from_domain_equals_dkim_domain":
            feature_2,

        # Link-level → aggregated
        "claimed_brand_equals_anchor":
            feature_3,

        "claimed_brand_equals_destination":
            feature_4,

        "authenticated_domain_equals_destination":
            feature_5,

        "claimed_or_authenticated_brand_equals_final_redirect":
            feature_6
    }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "sender_name":
            sender_name,

        "claimed_brand":
            claimed_brand,

        "from_domain":
            from_domain,

        "dkim_domain":
            dkim_domain,

        "number_of_links":
            number_of_links,

        "link_results":
            link_results
    }

    return features, metadata


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    eml_file = "sample.eml"

    bas_features, bas_metadata = (
        extract_bas_features(
            eml_file
        )
    )

    print(
        "\n========== BAS FEATURES ==========\n"
    )

    for key, value in bas_features.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\n========== BAS METADATA ==========\n"
    )

    print(
        "Sender name:",
        bas_metadata["sender_name"]
    )

    print(
        "Claimed brand:",
        bas_metadata["claimed_brand"]
    )

    print(
        "From domain:",
        bas_metadata["from_domain"]
    )

    print(
        "DKIM domain:",
        bas_metadata["dkim_domain"]
    )

    print(
        "Number of links:",
        bas_metadata["number_of_links"]
    )

    print("\n--- Individual Links ---\n")

    for i, link in enumerate(
        bas_metadata["link_results"],
        start=1
    ):

        print(f"LINK {i}")

        for key, value in link.items():
            print(
                f"  {key}: {value}"
            )

        print()