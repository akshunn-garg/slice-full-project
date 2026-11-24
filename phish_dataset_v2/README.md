# Phishing-email dataset builder (six exact feature modules)

`feature_modules/` = your six modules, byte-identical to the uploads:
`sts_features.py`, `bas_features.py`, `dms_features.py`, `los_features.py`, `rcs_features.py`,
`source_link_infra_features.py`. `build_dataset.py` only imports and calls them.

## Run on your own computer (needs direct internet; no proxy)

```bash
pip install -r requirements.txt
# Gmail API setup once: see the top of gmail_fetch.py (read-only scope)

# 1) 5-email test
python gmail_fetch.py --out private/inbox_eml --limit 5
python build_dataset.py --inbox-dir private/inbox_eml --limit 5 --out phishing_email_dataset_TEST.xlsx
python verify_dataset.py phishing_email_dataset_TEST.xlsx --expected 5 --id-map private/id_map.csv --recheck 5

# 2) full run: 5,000 newest Inbox emails (your Inbox has 16,491)
python gmail_fetch.py --out private/inbox_eml --limit 5000          # ~20-40 min, resumable
python build_dataset.py --inbox-dir private/inbox_eml --workers 8 --out phishing_email_dataset.xlsx
python verify_dataset.py phishing_email_dataset.xlsx --expected 5000 --id-map private/id_map.csv --recheck 5
# later, add phishing: --phishing-dir private/phish_eml (delete private/checkpoint.jsonl first)
```

`build_dataset.py` runs emails in parallel worker processes (`--workers`, default 8) and appends every
finished email to `private/checkpoint.jsonl`. If the run stops (sleep, network drop, Ctrl-C), run the
same command again: finished emails are not reprocessed. Delete the checkpoint to start over.
Expect several hours for 5,000 emails: the modules make up to ~8 network calls per link.

```bash
```

## Workbook
| Sheet | Content |
|---|---|
| Dataset | 1 row/email: `email_id`, `label`, `processing_status`, every email-level feature the six modules return (`STS__`, `BAS__`, `DMS__`, `LOS__`, `RCS__`, `SRC__` source-link, `INFRA__` infrastructure), non-personal module metadata (`*_meta__`), lookup status counts (`lookups_*`) |
| Link_Details | 1 row/link: `email_id`, `link_number`, `URL`, `destination`, `final_redirect_domain`, every per-link output of BAS/DMS/LOS/RCS/SRC/INFRA, per-link lookup status |
| Dataset_Stats | counts, duplicates, failures, feature counts per module |
| Processing_Errors | module exceptions, failed external lookups (per email/module/kind), environment warnings |
| Feature_Inventory | every column the modules actually produced: module, feature_name, data_type, email/link level |

Email-level values are the modules' own aggregation (unchanged). The SRC/INFRA split follows the module's
own two blocks ("SOURCE-LINK VECTOR" and "INFRASTRUCTURE").

## Lookup failures
The modules convert most lookup failures into 0 / -1 / original domain. The builder does not change that;
it records what happened (`lookups_failed_*`, `Link_Details.*status*`, `Processing_Errors`) so failures stay
identifiable.

## Safety and privacy
* The modules GET every link (BAS, DMS, RCS each) and open TLS connections to every link host.
  Unsubscribe-looking links are skipped by default (`--allow-unsubscribe-requests` to disable).
* No subject, body, anchor text, display names, addresses or Message-ID are written; URLs lose their
  query string and recipient identifiers (`--full-urls` keeps the query). SHA-256 hashes are only in the
  private `id_map.csv`. Keep `private/` out of git.
