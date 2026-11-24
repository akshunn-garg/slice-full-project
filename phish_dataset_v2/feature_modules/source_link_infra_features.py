import re
import ssl
import socket
import ipaddress
import requests

from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "zip", "mov",
    "tk", "ml", "ga", "cf", "gq",
    "work", "download", "stream", "live",
    "icu", "buzz", "monster", "cam"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def get_html(msg):

    if msg.is_multipart():

        for part in msg.walk():

            if part.get_content_type() == "text/html":

                try:
                    return part.get_content()
                except Exception:
                    pass

    elif msg.get_content_type() == "text/html":

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

        links.append(href)

    return links


# ============================================================
# URL-LEVEL FEATURES
# ============================================================

def hostname_length(url):

    hostname = urlparse(url).hostname or ""

    return len(hostname)


def path_length(url):

    path = urlparse(url).path or ""

    return len(path)


def dot_count(url):

    hostname = urlparse(url).hostname or ""

    return hostname.count(".")


def hyphen_count(url):

    hostname = urlparse(url).hostname or ""

    return hostname.count("-")


def slash_count(url):

    path = urlparse(url).path or ""

    return path.count("/")


def query_parameter_count(url):

    query = urlparse(url).query

    if not query:
        return 0

    return len(
        parse_qs(
            query,
            keep_blank_values=True
        )
    )


def digit_count(url):

    return sum(
        char.isdigit()
        for char in url
    )


def at_symbol(url):

    return int("@" in url)


def suspicious_tld(url):

    hostname = urlparse(url).hostname

    if not hostname:
        return 0

    parts = hostname.lower().split(".")

    if len(parts) < 2:
        return 0

    tld = parts[-1]

    return int(
        tld in SUSPICIOUS_TLDS
    )


def subdomain_depth(url):

    hostname = urlparse(url).hostname

    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 0
    except ValueError:
        pass

    parts = hostname.split(".")

    # Simple registered-domain assumption
    if len(parts) <= 2:
        return 0

    return len(parts) - 2


# ============================================================
# REPEATED BRAND TERMS
# ============================================================

def repeated_brand_terms(
    url,
    claimed_brand
):

    if not claimed_brand:
        return 0

    brand_words = re.findall(
        r'[a-zA-Z0-9]+',
        claimed_brand.lower()
    )

    if not brand_words:
        return 0

    url_text = url.lower()

    count = 0

    for word in brand_words:

        if len(word) < 3:
            continue

        count += len(
            re.findall(
                re.escape(word),
                url_text
            )
        )

    return count


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(
    url,
    claimed_brand
):

    return {

        "hostname_length":
            hostname_length(url),

        "path_length":
            path_length(url),

        "dot_count":
            dot_count(url),

        "hyphen_count":
            hyphen_count(url),

        "slash_count":
            slash_count(url),

        "query_parameter_count":
            query_parameter_count(url),

        "digit_count":
            digit_count(url),

        "at_symbol":
            at_symbol(url),

        "suspicious_tld":
            suspicious_tld(url),

        "repeated_brand_terms":
            repeated_brand_terms(
                url,
                claimed_brand
            ),

        "subdomain_depth":
            subdomain_depth(url)
    }


# ============================================================
# DNS TTL
# ============================================================

def get_dns_ttl(domain):

    try:

        import dns.resolver

        answer = dns.resolver.resolve(
            domain,
            "A"
        )

        return answer.rrset.ttl

    except Exception:

        return None


# ============================================================
# DNS RECORDS
# ============================================================

def get_dns_records(domain):

    result = {
        "A": False,
        "AAAA": False,
        "MX": False,
        "NS": False
    }

    try:

        import dns.resolver

        for record_type in result:

            try:

                dns.resolver.resolve(
                    domain,
                    record_type
                )

                result[record_type] = True

            except Exception:
                pass

    except Exception:
        pass

    return result


# ============================================================
# NAMESERVER AGE / STABILITY
# ============================================================

def get_nameserver_info(domain):

    nameservers = []

    try:

        import dns.resolver

        answers = dns.resolver.resolve(
            domain,
            "NS"
        )

        for answer in answers:

            ns = str(
                answer
            ).rstrip(".")

            nameservers.append(ns)

    except Exception:
        pass

    return {
        "nameserver_count":
            len(nameservers),

        "nameservers":
            nameservers
    }


# ============================================================
# ASN + HOSTING COUNTRY
# ============================================================

def get_ip_from_domain(domain):

    if not domain:
        return None

    try:

        return socket.gethostbyname(
            domain
        )

    except Exception:

        return None


def get_ip_metadata(ip):

    if not ip:
        return {
            "asn": None,
            "hosting_country": None
        }

    try:

        response = requests.get(
            f"https://ipinfo.io/{ip}/json",
            timeout=5
        )

        if response.status_code != 200:
            return {
                "asn": None,
                "hosting_country": None
            }

        data = response.json()

        organization = data.get(
            "org",
            ""
        )

        country = data.get(
            "country"
        )

        asn = None

        match = re.search(
            r'\bAS\d+\b',
            organization
        )

        if match:
            asn = match.group(0)

        return {
            "asn": asn,
            "hosting_country": country
        }

    except Exception:

        return {
            "asn": None,
            "hosting_country": None
        }


# ============================================================
# CERTIFICATE FEATURES
# ============================================================

def get_certificate_info(domain):

    result = {

        "certificate_existence": 0,

        "certificate_age": None,

        "certificate_SAN_count": 0
    }

    try:

        context = ssl.create_default_context()

        with socket.create_connection(
            (domain, 443),
            timeout=5
        ) as sock:

            with context.wrap_socket(
                sock,
                server_hostname=domain
            ) as ssock:

                cert = ssock.getpeercert()

                if cert:

                    result[
                        "certificate_existence"
                    ] = 1

                    # ----------------------------------------
                    # SAN count
                    # ----------------------------------------

                    san = cert.get(
                        "subjectAltName",
                        []
                    )

                    result[
                        "certificate_SAN_count"
                    ] = len(san)

                    # ----------------------------------------
                    # Certificate age
                    # ----------------------------------------

                    not_before = cert.get(
                        "notBefore"
                    )

                    if not_before:

                        try:

                            cert_date = datetime.strptime(
                                not_before,
                                "%b %d %H:%M:%S %Y %Z"
                            )

                            cert_date = cert_date.replace(
                                tzinfo=timezone.utc
                            )

                            result[
                                "certificate_age"
                            ] = (
                                datetime.now(
                                    timezone.utc
                                )
                                - cert_date
                            ).days

                        except Exception:
                            pass

    except Exception:
        pass

    return result


# ============================================================
# DOMAIN INFRASTRUCTURE FEATURES
# ============================================================

def extract_infrastructure_features(
    domain
):

    if not domain:

        return {

            "subdomain_depth": 0,

            "DNS_TTL": None,

            "nameserver_count": 0,

            "ASN": None,

            "hosting_country": None,

            "certificate_existence": 0,

            "certificate_age": None,

            "certificate_SAN_count": 0
        }

    # DNS
    dns_ttl = get_dns_ttl(
        domain
    )

    # Nameservers
    ns_info = get_nameserver_info(
        domain
    )

    # IP
    ip = get_ip_from_domain(
        domain
    )

    # ASN / country
    ip_info = get_ip_metadata(
        ip
    )

    # Certificate
    cert_info = get_certificate_info(
        domain
    )

    # Subdomain depth
    depth = subdomain_depth(
        "https://" + domain
    )

    return {

        "subdomain_depth":
            depth,

        "DNS_TTL":
            dns_ttl,

        "nameserver_count":
            ns_info[
                "nameserver_count"
            ],

        "ASN":
            ip_info["asn"],

        "hosting_country":
            ip_info[
                "hosting_country"
            ],

        "certificate_existence":
            cert_info[
                "certificate_existence"
            ],

        "certificate_age":
            cert_info[
                "certificate_age"
            ],

        "certificate_SAN_count":
            cert_info[
                "certificate_SAN_count"
            ]
    }


# ============================================================
# AGGREGATION HELPERS
# ============================================================

def average(values):

    values = [
        v for v in values
        if v is not None
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def maximum(values):

    values = [
        v for v in values
        if v is not None
    ]

    if not values:
        return 0.0

    return max(values)


def any_value(values):

    values = [
        v for v in values
        if v is not None
    ]

    if not values:
        return 0

    return int(
        any(v == 1 for v in values)
    )


# ============================================================
# COMPLETE SOURCE-LINK + INFRASTRUCTURE VECTOR
# ============================================================

def extract_source_link_vector(
    eml_path
):

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
    # Claimed brand
    # --------------------------------------------------------

    from_header = msg.get(
        "From",
        ""
    )

    match = re.match(
        r'^\s*"?([^"<]+)"?\s*<',
        from_header
    )

    if match:

        claimed_brand = (
            match.group(1).strip()
        )

    else:

        claimed_brand = None

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html = get_html(
        msg
    )

    links = extract_links(
        html
    )

    # --------------------------------------------------------
    # Per-link URL features
    # --------------------------------------------------------

    per_link = []

    for url in links:

        features = extract_url_features(
            url,
            claimed_brand
        )

        features["url"] = url

        per_link.append(
            features
        )

    # ========================================================
    # SOURCE-LINK VECTOR
    # ========================================================

    if per_link:

        avg_max_features = [
            "hostname_length",
            "path_length",
            "dot_count",
            "hyphen_count",
            "slash_count",
            "query_parameter_count",
            "digit_count",
            "repeated_brand_terms"
        ]

        binary_features = [
            "at_symbol",
            "suspicious_tld"
        ]

        vector = {}

        # --------------------------------------------
        # Average + max
        # --------------------------------------------

        for feature in avg_max_features:

            values = [
                x[feature]
                for x in per_link
            ]

            vector[
                feature + "_avg"
            ] = average(values)

            vector[
                feature + "_max"
            ] = maximum(values)

        # --------------------------------------------
        # Any / max
        # --------------------------------------------

        for feature in binary_features:

            values = [
                x[feature]
                for x in per_link
            ]

            vector[
                feature + "_any"
            ] = any_value(values)

    else:

        vector = {}

        for feature in [
            "hostname_length",
            "path_length",
            "dot_count",
            "hyphen_count",
            "slash_count",
            "query_parameter_count",
            "digit_count",
            "repeated_brand_terms"
        ]:

            vector[
                feature + "_avg"
            ] = 0.0

            vector[
                feature + "_max"
            ] = 0.0

        vector[
            "at_symbol_any"
        ] = 0

        vector[
            "suspicious_tld_any"
        ] = 0

    # ========================================================
    # INFRASTRUCTURE
    # ========================================================

    infrastructure_results = []

    for link in links:

        domain = (
            urlparse(link).hostname
        )

        if not domain:
            continue

        infrastructure = (
            extract_infrastructure_features(
                domain
            )
        )

        infrastructure_results.append(
            infrastructure
        )

    # --------------------------------------------------------
    # Infrastructure aggregation
    # --------------------------------------------------------

    if infrastructure_results:

        vector[
            "subdomain_depth_avg"
        ] = average([
            x["subdomain_depth"]
            for x in infrastructure_results
        ])

        vector[
            "subdomain_depth_max"
        ] = maximum([
            x["subdomain_depth"]
            for x in infrastructure_results
        ])

        vector[
            "DNS_TTL_avg"
        ] = average([
            x["DNS_TTL"]
            for x in infrastructure_results
        ])

        vector[
            "DNS_TTL_min"
        ] = min([
            x["DNS_TTL"]
            for x in infrastructure_results
            if x["DNS_TTL"] is not None
        ], default=0)

        vector[
            "nameserver_count_avg"
        ] = average([
            x["nameserver_count"]
            for x in infrastructure_results
        ])

        vector[
            "nameserver_count_max"
        ] = maximum([
            x["nameserver_count"]
            for x in infrastructure_results
        ])

        # --------------------------------------------
        # ASN
        # --------------------------------------------

        asns = set()

        for x in infrastructure_results:

            if x["ASN"]:
                asns.add(
                    x["ASN"]
                )

        vector[
            "unique_ASN_count"
        ] = len(asns)

        # --------------------------------------------
        # Hosting country
        # --------------------------------------------

        countries = set()

        for x in infrastructure_results:

            if x["hosting_country"]:
                countries.add(
                    x["hosting_country"]
                )

        vector[
            "unique_hosting_country_count"
        ] = len(countries)

        # --------------------------------------------
        # Certificate existence
        # --------------------------------------------

        vector[
            "certificate_existence_any"
        ] = any_value([
            x["certificate_existence"]
            for x in infrastructure_results
        ])

        # --------------------------------------------
        # Certificate age
        # --------------------------------------------

        vector[
            "certificate_age_avg"
        ] = average([
            x["certificate_age"]
            for x in infrastructure_results
        ])

        vector[
            "certificate_age_min"
        ] = min([
            x["certificate_age"]
            for x in infrastructure_results
            if x["certificate_age"] is not None
        ], default=0)

        # --------------------------------------------
        # SAN count
        # --------------------------------------------

        vector[
            "certificate_SAN_count_avg"
        ] = average([
            x["certificate_SAN_count"]
            for x in infrastructure_results
        ])

        vector[
            "certificate_SAN_count_max"
        ] = maximum([
            x["certificate_SAN_count"]
            for x in infrastructure_results
        ])

    else:

        vector.update({

            "subdomain_depth_avg": 0.0,
            "subdomain_depth_max": 0.0,

            "DNS_TTL_avg": 0.0,
            "DNS_TTL_min": 0.0,

            "nameserver_count_avg": 0.0,
            "nameserver_count_max": 0.0,

            "unique_ASN_count": 0,

            "unique_hosting_country_count": 0,

            "certificate_existence_any": 0,

            "certificate_age_avg": 0.0,
            "certificate_age_min": 0.0,

            "certificate_SAN_count_avg": 0.0,
            "certificate_SAN_count_max": 0.0
        })

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "vector": vector,
        "per_link": per_link,
        "infrastructure": infrastructure_results,
        "claimed_brand": claimed_brand,
        "number_of_links": len(links)
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    eml_path = "sample.eml"

    result = extract_source_link_vector(
        eml_path
    )

    print(
        "\n========== SOURCE-LINK + INFRASTRUCTURE VECTOR =========="
    )

    for feature, value in result["vector"].items():

        print(
            f"{feature}: {value}"
        )

    print(
        "\nNumber of links:",
        result["number_of_links"]
    )