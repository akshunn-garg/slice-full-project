"""
Reinforcement-learning combiner for the two SLiCE outputs (deployment-time learning).

    Output 1  STS_rule  - rule-based source trust (slice_models.sts_rules), never trained.
    Output 2  link_prob - learned link risk (BAS, DMS, LOS, RCS + SRC/INFRA -> final link model).

The combiner is a contextual bandit: for each email it takes an action (1 = flag as phishing,
0 = allow), receives a reward from feedback (analyst / user: +reward if the action was right,
-cost if wrong) and updates how much it trusts each output. It never learns from the training
dataset, so "legit mail always passes authentication" in that dataset cannot leak into it.

Model:  p(phishing) = sigmoid(b + a * logit(link_prob) + w * (STS_risk - 0.5)),  STS_risk = 1 - STS_rule
  * starts at the tested fixed combination (b=0, a=1, w=3), so behaviour before any
    feedback equals the evaluated model;
  * a >= 0 and 0 <= w <= w_max (monotonic: more risk can never lower the score, and STS can
    never dominate beyond w_max);
  * Bayesian online logistic regression with a diagonal Gaussian posterior (Laplace
    approximation, Chapelle & Li 2011) updated in small batches; actions by Thompson sampling
    (explore=True) or by the posterior mean (explore=False);
  * decision threshold from costs: flag if p >= cost_false_alarm / (cost_false_alarm + cost_missed).

With binary actions the reward reveals the true label (flag + positive reward -> phishing, flag +
negative reward -> legit, and the reverse for allow), so every feedback event is informative.

    from slice_models.rl_combiner import SliceRLCombiner
    rl = SliceRLCombiner()                                     # starts at link + 3 * STS
    act, info = rl.decide(link_prob, sts_rule)                 # 1 = flag, 0 = allow
    rl.feedback(link_prob, sts_rule, act, reward)              # reward > 0 right, < 0 wrong
    rl.save("models/final3/rl_combiner_state.json"); rl = SliceRLCombiner.load(...)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

NAMES = ("bias", "link_logit_weight", "sts_weight")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


@dataclass
class SliceRLCombiner:
    mean: list = field(default_factory=lambda: [0.0, 1.0, 3.0])      # b, a, w  (start = tested fixed combination)
    precision: list = field(default_factory=lambda: [2.0, 20.0, 5.0])  # prior strength per parameter
    w_max: float = 4.0
    cost_missed_phishing: float = 5.0
    cost_false_alarm: float = 1.0
    reward_correct: float = 1.0
    batch_size: int = 50
    explore: bool = True
    seed: int = 42
    n_feedback: int = 0
    history: list = field(default_factory=list)                      # posterior mean after each batch
    _buffer: list = field(default_factory=list)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed + self.n_feedback)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def context(link_prob, sts_rule):
        link_prob, sts_rule = np.atleast_1d(link_prob), np.atleast_1d(sts_rule)
        return np.column_stack([np.ones(len(link_prob)), _logit(link_prob), (1 - sts_rule) - 0.5])

    def _bounds(self):
        return [(None, None), (0.0, None), (0.0, self.w_max)]

    def _clip(self, theta):
        lo = np.array([-np.inf, 0.0, 0.0])
        hi = np.array([np.inf, np.inf, self.w_max])
        return np.clip(theta, lo, hi)

    @property
    def threshold(self) -> float:
        return self.cost_false_alarm / (self.cost_false_alarm + self.cost_missed_phishing)

    # ---------------------------------------------------------------- decision
    def decide(self, link_prob, sts_rule, explore: bool | None = None):
        """Return (actions, info). actions: 1 = flag as phishing, 0 = allow."""
        x = self.context(link_prob, sts_rule)
        explore = self.explore if explore is None else explore
        m, q = np.array(self.mean), np.array(self.precision)
        theta = self._clip(self._rng.normal(m, 1 / np.sqrt(q))) if explore else m
        z_parts = x * theta
        p = _sigmoid(z_parts.sum(axis=1))
        actions = (p >= self.threshold).astype(int)
        info = {"p_phishing": p, "threshold": self.threshold,
                "contribution_link": z_parts[:, 1], "contribution_sts": z_parts[:, 2], "bias": z_parts[:, 0],
                "parameters_used": dict(zip(NAMES, theta.round(4).tolist()))}
        return actions, info

    # ---------------------------------------------------------------- learning
    def feedback(self, link_prob, sts_rule, action, reward):
        """Record one feedback event per email; the posterior is updated every batch_size events."""
        x = self.context(link_prob, sts_rule)
        action, reward = np.atleast_1d(action).astype(int), np.atleast_1d(reward).astype(float)
        labels = np.where(reward > 0, action, 1 - action)        # reward reveals the true label
        for xi, yi in zip(x, labels):
            self._buffer.append((xi.tolist(), int(yi)))
            self.n_feedback += 1
            if len(self._buffer) >= self.batch_size:
                self._update()

    def reward_for(self, action, label):
        """Reward the environment would give (used by simulations; deployment gets it from people)."""
        action, label = np.atleast_1d(action), np.atleast_1d(label)
        r = np.where(action == label, self.reward_correct, 0.0)
        r = np.where((action == 0) & (label == 1), -self.cost_missed_phishing, r)
        r = np.where((action == 1) & (label == 0), -self.cost_false_alarm, r)
        return r

    def _update(self):
        X = np.array([b[0] for b in self._buffer])
        y = np.array([b[1] for b in self._buffer])
        m, q = np.array(self.mean), np.array(self.precision)

        def objective(theta):
            z = X @ theta
            nll = np.sum(np.logaddexp(0, z) - y * z)
            prior = 0.5 * np.sum(q * (theta - m) ** 2)
            grad = X.T @ (_sigmoid(z) - y) + q * (theta - m)
            return nll + prior, grad

        res = minimize(objective, m, jac=True, method="L-BFGS-B", bounds=self._bounds())
        theta = self._clip(res.x)
        p = _sigmoid(X @ theta)
        self.mean = theta.tolist()
        self.precision = (q + (p * (1 - p)) @ (X ** 2)).tolist()
        self.history.append({"n_feedback": self.n_feedback, **dict(zip(NAMES, theta.round(4).tolist()))})
        self._buffer = []

    # ------------------------------------------------------------- persistence
    def save(self, path):
        state = {k: v for k, v in asdict(self).items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(state, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
