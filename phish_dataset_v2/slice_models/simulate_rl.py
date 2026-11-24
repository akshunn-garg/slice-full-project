"""
Replay simulation of the RL combiner (a demonstration, not deployment evidence).

    python -m slice_models.simulate_rl            # writes reports/slice_v4/rl_simulation*.csv

Stream: the val + test emails of slice_dataset_v4 in arrival order (the link model never trained on
them). For each email the combiner acts (flag / allow) and receives the reward it would get from
feedback, using the dataset label. Four scenarios change the emails' authentication evidence so that
STS becomes misleading in ways the training data never showed:
  observed            - authentication as it is (legit almost always passes)
  legit_auth_fail     - 30% of legit get mailing-list style failures (SPF fail, DKIM broken, DMARC fail)
  compromised_senders - 30% of phishing get perfect authentication (SPF/DKIM/DMARC pass, aligned)
  both                - both at once
Each scenario compares the fixed combination (link + 3 * STS, no learning) with the RL combiner,
over several random seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.rl_combiner import SliceRLCombiner
from slice_models.slice_final_predict import predict_final
from slice_models.sts_rules import sts_rule_score

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ("observed", "legit_auth_fail", "compromised_senders", "both")


def perturb(d: pd.DataFrame, scenario: str, rng) -> pd.DataFrame:
    x = d.copy()
    legit, phish = np.flatnonzero(x.label.values == 0), np.flatnonzero(x.label.values == 1)
    if scenario in ("legit_auth_fail", "both"):
        idx = x.index[rng.choice(legit, int(0.3 * len(legit)), replace=False)]
        x.loc[idx, ["STS_meta__spf_result", "STS_meta__dkim_result", "STS_meta__dmarc_result"]] = ["fail", "fail", "fail"]
        x.loc[idx, "STS__dkim_aligned"] = 0
    if scenario in ("compromised_senders", "both"):
        idx = x.index[rng.choice(phish, int(0.3 * len(phish)), replace=False)]
        x.loc[idx, ["STS_meta__spf_result", "STS_meta__dkim_result", "STS_meta__dmarc_result"]] = ["pass", "pass", "pass"]
        x.loc[idx, "STS__dkim_aligned"] = 1
    return x


def run(link_prob, sts, labels, learn: bool, seed: int, explore: bool = True):
    rl = SliceRLCombiner(seed=seed, explore=learn and explore)
    acts = np.zeros(len(labels), dtype=int)
    for i in range(len(labels)):
        a, _ = rl.decide(link_prob[i], sts[i])
        acts[i] = a[0]
        if learn:
            rl.feedback(link_prob[i], sts[i], a, rl.reward_for(a, labels[i]))
    return acts, rl


def metrics(acts, labels, rl, part):
    a, y = acts[part], labels[part]
    return {"reward": float(rl.reward_for(a, y).sum()),
            "false_alarm_rate": float(((a == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "missed_phishing_rate": float(((a == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))}


def main():
    d = pd.read_excel(ROOT / "slice_dataset_v4.xlsx", sheet_name="Dataset")
    d = d[d.prov__time_split.isin(["val", "test"])].sort_values("prov__arrival_time").reset_index(drop=True)
    link_prob = predict_final(d)["FINAL_link_prob"].to_numpy()
    labels = d.label.to_numpy()
    second_half = np.arange(len(d)) >= len(d) // 2          # after the learner has seen feedback
    rows, traj = [], []
    for scenario in SCENARIOS:
        for seed in range(5):
            x = perturb(d, scenario, np.random.default_rng(seed))
            sts = sts_rule_score(x)["STS_rule"].to_numpy()
            for policy, learn, explore in (("fixed (link + 3*STS)", False, False),
                                           ("RL, Thompson sampling", True, True),
                                           ("RL, greedy (no exploration)", True, False)):
                acts, rl = run(link_prob, sts, labels, learn, seed, explore)
                rows.append({"scenario": scenario, "seed": seed, "policy": policy,
                             **{f"all_{k}": v for k, v in metrics(acts, labels, rl, np.ones(len(d), bool)).items()},
                             **{f"second_half_{k}": v for k, v in metrics(acts, labels, rl, second_half).items()},
                             "final_bias": rl.mean[0], "final_link_weight": rl.mean[1], "final_sts_weight": rl.mean[2]})
                if learn and explore and seed == 0:
                    traj += [{"scenario": scenario, **h} for h in rl.history]
    res = pd.DataFrame(rows)
    out = ROOT / "reports" / "slice_v4"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "rl_simulation_runs.csv", index=False)
    pd.DataFrame(traj).to_csv(out / "rl_simulation_weight_trajectory.csv", index=False)
    summary = (res.groupby(["scenario", "policy"], sort=False)
               [["second_half_false_alarm_rate", "second_half_missed_phishing_rate", "all_reward", "final_sts_weight"]]
               .mean().round(4))
    summary.to_csv(out / "rl_simulation_summary.csv")
    print(f"stream: {len(d)} emails (val+test, arrival order), costs: missed phishing 5, false alarm 1, correct +1")
    print(summary.to_string())


if __name__ == "__main__":
    main()
