# slice_dataset_v3 build

Builds `slice_dataset_v3.xlsx` (+ `slice_dataset_v3_core.csv`) from the two v2 workbooks without re-running the feature modules.

```bash
# 1. GitHub tree listing of phishing_pot, then the raw .eml files (cache dir outside the project)
curl -s "https://api.github.com/repos/rf-peixoto/phishing_pot/git/trees/main?recursive=1" -o tree.json
python -I build_v3/pot_download.py tree.json <pot_raw_dir>
# 2. provenance + receiver-normalised received-chain features + replay of the builder's phish_ id assignment
../.venv/Scripts/python.exe build_v3/provenance.py . <pot_raw_dir> tree.json provenance.csv
# 3. assemble (validates the id mapping against stored values first)
../.venv/Scripts/python.exe build_v3/assemble_v3.py . provenance.csv
# 4. optional diagnostics (needs lightgbm)
python build_v3/diagnose_v3.py slice_dataset_v3.xlsx
```

What changes vs v2: `STS__received_hops` and `STS__unusual_received_chain` are recomputed with
`sts_features.analyze_received_chain` after removing the receiving provider's own leading `Received`
headers (originals kept as `STS_raw__*`). All other module features are copied unchanged.
`prov__*` columns add source, receiver, arrival time, Microsoft delivery folder, sender org, and the
`prov__core` subset with a time-ordered `prov__time_split`. See the `Column_Guide` sheet.
