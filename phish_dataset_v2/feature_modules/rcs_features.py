import re
import requests
import ipaddress
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
from bs4 import BeautifulSoup


# ============================================================
# 1. KNOWN SHORTENERS
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
# 2. DOMAIN HELPERS
# ============================================================

def organizational_domain(domain):

    if not domain:
        return None

    domain = domain.lower().strip(".")

    try:
        ipaddress.ip_address(domain)
        return domain
    except ValueError:
        pass

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


def get_hostname(url):

    try:
        return urlparse(url).hostname
    except Exception:
        return None


# ============================================================
# 3. EXTRACT EMAIL INFORMATION
# ============================================================

def get_from_domain(msg):

    from_header = msg.get("From", "")

    match = re.search(
        r'@([A-Za-z0-9.-]+)',
        from_header
    )

    if match:
        return match.group(1).lower()

    return None


def get_dkim_domain(msg):

    auth_results = msg.get_all(
        "Authentication-Results",
        []
    )

    for header in auth_results:

        match = re.search(
            r'dkim\s*=\s*(?:pass|fail|softfail|neutral|none|temperror|permerror)'
            r'.*?\bd=([A-Za-z0-9.-]+)',
            header,
            re.I
        )

        if match:
            return match.group(1).lower()

    # Fallback
    dkim_signature = msg.get(
        "DKIM-Signature",
        ""
    )

    match = re.search(
        r'\bd=([A-Za-z0-9.-]+)',
        dkim_signature,
        re.I
    )

    if match:
        return match.group(1).lower()

    return None


def extract_claimed_brand(msg):

    from_header = msg.get(
        "From",
        ""
    )

    match = re.match(
        r'^\s*"?([^"<]+)"?\s*<',
        from_header
    )

    if match:
        return match.group(1).strip()

    return None


# ============================================================
# 4. HTML
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

        anchor = tag.get_text(
            " ",
            strip=True
        )

        links.append({
            "url": href,
            "anchor": anchor
        })

    return links


# ============================================================
# 5. VISIBLE URL
# ============================================================

def extract_visible_url(anchor):

    if not anchor:
        return None

    match = re.search(
        r'https?://[^\s<>"\']+',
        anchor,
        re.I
    )

    if match:
        return match.group(0).rstrip(
            ".,);]"
        )

    return None


# ============================================================
# 6. FOLLOW REDIRECT CHAIN
# ============================================================

def get_redirect_chain(url):

    """
    Returns the complete redirect chain.

    Example:

    shortener.com
          ↓
    tracker.com
          ↓
    google.com

    """

    try:

        response = requests.get(
            url,
            allow_redirects=True,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        chain = []

        # Original URL
        chain.append(url)

        # Intermediate/final responses
        for resp in response.history:

            chain.append(
                resp.url
            )

            location = resp.headers.get(
                "Location"
            )

            if location:
                chain.append(
                    location
                )

        # Final URL
        if response.url not in chain:
            chain.append(
                response.url
            )

        # Remove duplicates while preserving order
        cleaned = []

        for item in chain:

            if item and item not in cleaned:
                cleaned.append(item)

        return cleaned

    except Exception:

        return [url]


# ============================================================
# 7. REDIRECT DOMAINS
# ============================================================

def get_redirect_domains(chain):

    domains = []

    for url in chain:

        domain = registered_domain_from_url(
            url
        )

        if domain and domain not in domains:
            domains.append(domain)

    return domains


# ============================================================
# 8. COUNT REDIRECT HOPS — 5 LABELS
# ============================================================

def redirect_hop_label(hop_count):

    # 0 = no redirect
    # 1 = one hop
    # 2 = two hops
    # 3 = three hops
    # 4 = four or more hops

    if hop_count == 0:
        return 0

    elif hop_count == 1:
        return 1

    elif hop_count == 2:
        return 2

    elif hop_count == 3:
        return 3

    else:
        return 4


# ============================================================
# 9. REDIRECT DOMAIN CONSISTENCY
# ============================================================

def redirect_domain_consistency(
    redirect_domains,
    expected_domain
):

    if not redirect_domains:
        return 0.0

    if not expected_domain:
        return 0.0

    expected_domain = organizational_domain(
        expected_domain
    )

    matches = 0

    for domain in redirect_domains:

        if organizational_domain(domain) == expected_domain:
            matches += 1

    return matches / len(redirect_domains)


# ============================================================
# 10. FINAL DOMAIN == VISIBLE DOMAIN
# ============================================================

def final_domain_equals_visible(
    final_domain,
    visible_url
):

    visible_domain = registered_domain_from_url(
        visible_url
    )

    if not final_domain or not visible_domain:
        return 0

    return int(
        organizational_domain(final_domain)
        ==
        organizational_domain(visible_domain)
    )


# ============================================================
# 11. EXPECTED HTTPS BUT FINAL HTTP
# ============================================================

def expected_https_final_http(
    expected_domain,
    original_url,
    final_url
):

    if not expected_domain:
        return 0

    original_scheme = urlparse(
        original_url
    ).scheme.lower()

    final_scheme = urlparse(
        final_url
    ).scheme.lower()

    return int(
        original_scheme == "https"
        and final_scheme == "http"
    )


# ============================================================
# 12. DOMAIN REGISTRATION AGE
# ============================================================

def get_domain_creation_date(domain):

    if not domain:
        return None

    try:

        response = requests.get(
            f"https://rdap.org/domain/{domain}",
            timeout=8,
            headers={
                "User-Agent":
                "RCS-feature-extractor/1.0"
            }
        )

        if response.status_code != 200:
            return None

        data = response.json()

        events = data.get(
            "events",
            []
        )

        for event in events:

            if event.get(
                "eventAction"
            ) == "registration":

                date_string = event.get(
                    "eventDate"
                )

                if date_string:

                    return datetime.fromisoformat(
                        date_string.replace(
                            "Z",
                            "+00:00"
                        )
                    )

    except Exception:
        pass

    return None


def domain_age_days(domain):

    creation_date = get_domain_creation_date(
        domain
    )

    if not creation_date:
        return None

    now = datetime.now(
        timezone.utc
    )

    return max(
        0,
        (now - creation_date).days
    )


# ============================================================
# 13. FINAL DOMAIN REGISTRATION AGE — 5 LABELS
# ============================================================

def registration_age_label(age_days):

    if age_days is None:
        return 0

    # 0 = <= 7 days
    # 1 = 8–30 days
    # 2 = 31–90 days
    # 3 = 91–365 days
    # 4 = > 365 days

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
# 14. FINAL IP
# ============================================================

def resolve_domain_to_ip(domain):

    if not domain:
        return None

    try:

        import socket

        return socket.gethostbyname(
            domain
        )

    except Exception:

        return None


# ============================================================
# 15. IP ABUSE INTELLIGENCE
# ============================================================

def ip_in_abuse_intelligence(ip):

    """
    Placeholder for an abuse-intelligence API.

    Set your API here when you choose the provider.

    Returns:
        1 = IP found in abuse intelligence
        0 = not found / unavailable
    """

    if not ip:
        return 0

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT blindly call a random API.
    #
    # Replace this section with your selected provider.
    # --------------------------------------------------------

    return 0


# ============================================================
# 16. UNIQUE ASNs AND COUNTRIES
# ============================================================

def get_ip_metadata(ip):

    """
    Retrieves ASN and country for an IP.

    Uses ipinfo.io as an example.

    You can replace this with your preferred
    infrastructure-intelligence provider.
    """

    if not ip:
        return None, None

    try:

        response = requests.get(
            f"https://ipinfo.io/{ip}/json",
            timeout=5,
            headers={
                "User-Agent":
                "RCS-feature-extractor/1.0"
            }
        )

        if response.status_code != 200:
            return None, None

        data = response.json()

        org = data.get(
            "org",
            ""
        )

        country = data.get(
            "country"
        )

        # Example:
        # "AS15169 Google LLC"
        asn = None

        match = re.search(
            r'\bAS\d+\b',
            org
        )

        if match:
            asn = match.group(0)

        return asn, country

    except Exception:

        return None, None


# ============================================================
# 17. SHORTENER START + WRONG FINAL DOMAIN
# ============================================================

def shortener_start_wrong_final(
    original_url,
    final_domain,
    expected_domain
):

    shortener = is_known_shortener(
        original_url
    )

    if not shortener:
        return 0

    if not final_domain:
        return 0

    if not expected_domain:
        return 0

    return int(
        organizational_domain(final_domain)
        !=
        organizational_domain(expected_domain)
    )


def is_known_shortener(url):

    domain = registered_domain_from_url(
        url
    )

    if not domain:
        return 0

    return int(
        domain.lower()
        in KNOWN_SHORTENERS
    )


# ============================================================
# 18. ANALYZE ONE LINK
# ============================================================

def analyze_link(
    url,
    anchor,
    expected_domain
):

    # Visible URL
    visible_url = extract_visible_url(
        anchor
    )

    # Redirect chain
    chain = get_redirect_chain(
        url
    )

    # Redirect domains
    redirect_domains = get_redirect_domains(
        chain
    )

    # Final URL
    final_url = (
        chain[-1]
        if chain
        else url
    )

    final_domain = registered_domain_from_url(
        final_url
    )

    # Number of redirect hops
    # Number of transitions:
    # A -> B = 1 hop
    hop_count = max(
        0,
        len(chain) - 1
    )

    # Final IP
    final_ip = resolve_domain_to_ip(
        final_domain
    )

    # ASN / countries
    asns = set()
    countries = set()

    if final_ip:

        asn, country = get_ip_metadata(
            final_ip
        )

        if asn:
            asns.add(asn)

        if country:
            countries.add(country)

    # --------------------------------------------------------
    # RCS FEATURES
    # --------------------------------------------------------

    f1 = redirect_hop_label(
        hop_count
    )

    f2 = redirect_domain_consistency(
        redirect_domains,
        expected_domain
    )

    f3 = final_domain_equals_visible(
        final_domain,
        visible_url
    )

    f4 = expected_https_final_http(
        expected_domain,
        url,
        final_url
    )

    age_days = domain_age_days(
        final_domain
    )

    f5 = registration_age_label(
        age_days
    )

    f6 = ip_in_abuse_intelligence(
        final_ip
    )

    f7_asn_count = len(asns)

    f7_country_count = len(countries)

    f8 = shortener_start_wrong_final(
        url,
        final_domain,
        expected_domain
    )

    return {

        "url": url,

        "anchor": anchor,

        "redirect_chain": chain,

        "redirect_domains": redirect_domains,

        "redirect_hops": hop_count,

        "redirect_hop_label": f1,

        "redirect_domain_consistency": f2,

        "final_url": final_url,

        "final_domain": final_domain,

        "visible_url": visible_url,

        "final_domain_equals_visible": f3,

        "expected_https_final_http": f4,

        "final_domain_registration_age_days":
            age_days,

        "final_domain_registration_age_label":
            f5,

        "final_ip": final_ip,

        "final_ip_in_abuse_intelligence":
            f6,

        "unique_asns":
            f7_asn_count,

        "unique_countries":
            f7_country_count,

        "shortener_start_final_domain_wrong":
            f8
    }


# ============================================================
# 19. MAIN RCS EXTRACTION
# ============================================================

def extract_rcs_features(eml_path):

    # --------------------------------------------------------
    # Read email
    # --------------------------------------------------------

    with open(
        eml_path,
        "rb"
    ) as f:

        msg = BytesParser(
            policy=policy.default
        ).parse(f)

    # --------------------------------------------------------
    # Expected domain
    # --------------------------------------------------------

    from_domain = get_from_domain(
        msg
    )

    dkim_domain = get_dkim_domain(
        msg
    )

    # Prefer DKIM domain
    if dkim_domain:

        expected_domain = (
            organizational_domain(
                dkim_domain
            )
        )

    else:

        expected_domain = (
            organizational_domain(
                from_domain
            )
            if from_domain
            else None
        )

    # --------------------------------------------------------
    # HTML links
    # --------------------------------------------------------

    html = get_html(msg)

    links = extract_links(
        html
    )

    # --------------------------------------------------------
    # Analyze each link
    # --------------------------------------------------------

    link_results = []

    for i, link in enumerate(
        links,
        start=1
    ):

        result = analyze_link(
            link["url"],
            link["anchor"],
            expected_domain
        )

        result["link_number"] = i

        link_results.append(
            result
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    n = len(link_results)

    if n == 0:

        features = {

            "redirect_hop_label": 0.0,

            "redirect_domain_consistency": 0.0,

            "final_domain_equals_visible": 0.0,

            "expected_https_final_http": 0.0,

            "final_domain_registration_age_label": 0.0,

            "final_ip_in_abuse_intelligence": 0.0,

            "unique_asns": 0.0,

            "unique_countries": 0.0,

            "shortener_start_final_domain_wrong": 0.0
        }

    else:

        features = {

            # 5-label feature → average
            "redirect_hop_label":
                sum(
                    x["redirect_hop_label"]
                    for x in link_results
                ) / n,

            # Continuous ratio → average
            "redirect_domain_consistency":
                sum(
                    x["redirect_domain_consistency"]
                    for x in link_results
                ) / n,

            # Binary → proportion
            "final_domain_equals_visible":
                sum(
                    x["final_domain_equals_visible"]
                    for x in link_results
                ) / n,

            # Binary → proportion
            "expected_https_final_http":
                sum(
                    x["expected_https_final_http"]
                    for x in link_results
                ) / n,

            # 5-label → average
            "final_domain_registration_age_label":
                sum(
                    x[
                        "final_domain_registration_age_label"
                    ]
                    for x in link_results
                ) / n,

            # Binary → proportion
            "final_ip_in_abuse_intelligence":
                sum(
                    x[
                        "final_ip_in_abuse_intelligence"
                    ]
                    for x in link_results
                ) / n,

            # Continuous → average
            "unique_asns":
                sum(
                    x["unique_asns"]
                    for x in link_results
                ) / n,

            # Continuous → average
            "unique_countries":
                sum(
                    x["unique_countries"]
                    for x in link_results
                ) / n,

            # Binary → proportion
            "shortener_start_final_domain_wrong":
                sum(
                    x[
                        "shortener_start_final_domain_wrong"
                    ]
                    for x in link_results
                ) / n
        }

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "from_domain":
            from_domain,

        "dkim_domain":
            dkim_domain,

        "expected_domain":
            expected_domain,

        "number_of_links":
            n,

        "link_results":
            link_results
    }

    return features, metadata


# ============================================================
# 20. RUN
# ============================================================

if __name__ == "__main__":

    eml_path = "sample.eml"

    features, metadata = extract_rcs_features(
        eml_path
    )

    print(
        "\n========== RCS FEATURES =========="
    )

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )

    print(
        "\n========== RCS METADATA =========="
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

    print(
        "\n========== INDIVIDUAL LINKS =========="
    )

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
            "Redirect chain:",
            link["redirect_chain"]
        )

        print(
            "Redirect domains:",
            link["redirect_domains"]
        )

        print(
            "Redirect hops:",
            link["redirect_hops"]
        )

        print(
            "Redirect hop label:",
            link["redirect_hop_label"]
        )

        print(
            "Redirect domain consistency:",
            link["redirect_domain_consistency"]
        )

        print(
            "Visible URL:",
            link["visible_url"]
        )

        print(
            "Final URL:",
            link["final_url"]
        )

        print(
            "Final domain:",
            link["final_domain"]
        )

        print(
            "Final domain == visible:",
            link["final_domain_equals_visible"]
        )

        print(
            "Expected HTTPS → Final HTTP:",
            link["expected_https_final_http"]
        )

        print(
            "Final domain age:",
            link[
                "final_domain_registration_age_days"
            ]
        )

        print(
            "Final domain age label:",
            link[
                "final_domain_registration_age_label"
            ]
        )

        print(
            "Final IP:",
            link["final_ip"]
        )

        print(
            "IP in abuse intelligence:",
            link[
                "final_ip_in_abuse_intelligence"
            ]
        )

        print(
            "Unique ASNs:",
            link["unique_asns"]
        )

        print(
            "Unique countries:",
            link["unique_countries"]
        )

        print(
            "Shortener start + wrong final:",
            link[
                "shortener_start_final_domain_wrong"
            ]
        )