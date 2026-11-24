"""LLM classification of the text problems (zero-shot / few-shot) via API, on the SAME folds.

Providers:
  anthropic   - Claude (e.g. claude-opus-5-5); key: environment variable ANTHROPIC_API_KEY
  openrouter  - e.g. Qwen (OpenAI-compatible API); key: environment variable OPENROUTER_API_KEY
  mock        - fake model to check that the script runs (no API calls)

Protocol (fixed BEFORE the full run):
  - prompt: fixed (PROMPT_VERSION below), definitions of the 4 areas, answer ends with the area name;
  - temperature 0 (if the model does not accept it, the call is made without it and this is recorded);
  - zero-shot: no examples (the paper uses zero-shot only);
  - few-shot: K examples per area, sampled (seed 42) FROM THE TRAINING PART of the same fold
    (folds.csv, repetition 0) -> no test problem is ever an example for itself;
  - every answer is written to responses.jsonl immediately -> after an interruption the same command resumes;
  - invalid answer: one follow-up request, then the label INVALID (counted as an error).

Commands (from the repository root):
    python src/models/llm_classify_text.py --provider mock --limit 40              # dry run
    python src/models/llm_classify_text.py --provider anthropic --model claude-opus-5-5 --limit 50 --pilot
    python src/models/llm_classify_text.py --provider anthropic --model claude-opus-5-5
    python src/models/llm_classify_text.py --provider anthropic --model claude-opus-5-5 --shots 2
    python src/models/llm_classify_text.py --provider openrouter --model qwen/qwen3.6-27b --or-host Alibaba

BATCH mode (Anthropic Message Batches API; results usually within 1 h, at most 24 h):
    python src/models/llm_classify_text.py --provider anthropic --model claude-opus-5-5 --batch submit
    python src/models/llm_classify_text.py --provider anthropic --model claude-opus-5-5 --batch status
    python src/models/llm_classify_text.py --provider anthropic --model claude-opus-5-5 --batch collect
  submit sends all unfinished problems in one batch; collect downloads the results, re-sends
  invalid answers through the standard API, then computes the metrics (same output as without batch).

--pilot: random stratified sample of --limit problems (for prompt development);
         pilot results go to a separate folder and are NOT used in the paper.

Output: reports/ml/text_v2/llm_<provider>_<model>_<k>shot[/pilot]/
    responses.jsonl, predictions.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
FOLDS = REPO_ROOT / "reports" / "ml" / "text_v2" / "folds.csv"
OOF_LINEAR = REPO_ROOT / "reports" / "ml" / "text_v2" / "oof_predictions_rep0.csv"
OUT_ROOT = REPO_ROOT / "reports" / "ml" / "text_v2"
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Kombinatorika"]

PROMPT_VERSION = "text-v2"
SYSTEM_PROMPT = """You classify Serbian high-school mathematics competition problems by mathematical area.
The problem statement is in Serbian (Latin script) and may contain LaTeX.
Choose exactly ONE of the following four areas:
- Algebra: equations, inequalities, functions, polynomials, sequences and series, algebraic identities, real and complex numbers.
- Geometrija: plane and solid geometry, triangles, circles, polygons, angles, lengths, areas, volumes, geometric constructions and transformations.
- Teorija brojeva: integers, divisibility, primes, remainders and congruences, digits, Diophantine equations, gcd and lcm.
- Kombinatorika: counting, arrangements, combinatorial games, colourings, graphs, invariants, pigeonhole principle, tables and boards.
If a problem touches several areas, choose the one that is central to solving it.
You may think briefly first. Your answer must end with one final line of exactly this form:
Oblast: <area>
where <area> is one of: Algebra, Geometrija, Teorija brojeva, Kombinatorika."""


# --------------------------------------------------------------------------------------------
def normalize_label(text: str) -> str | None:
    """Reads the final line 'Oblast: <area>'; without it, only an answer that is exactly one area is accepted."""
    t0 = unicodedata.normalize("NFC", text or "")
    m = re.findall(r"oblast\s*[:\-]\s*\**\s*([^\n*]+)", t0, flags=re.I)
    if m:
        t0 = m[-1]
    t = t0.strip().strip(" .:*\"'`\n").lower()
    t = re.sub(r"\s+", " ", t)
    for c in CLASSES:
        if t == c.lower():
            return c
    if m:  # a short suffix is allowed in the final 'Oblast:' line (e.g. "Algebra (borderline)")
        hits = [c for c in CLASSES if c.lower() in t]
        return hits[0] if len(hits) == 1 else None
    return None  # without an 'Oblast:' line free text is NOT parsed (an area name may be mentioned in passing)


class Provider:
    def __init__(self, name: str, model: str, max_tokens: int = 1024, extra: dict | None = None):
        self.name, self.model, self.max_tokens = name, model, max_tokens
        self.extra = extra or {}
        self.use_temperature = True
        self.temperature_note = "temperature=0"
        if name == "anthropic":
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY is not set. In PowerShell: $env:ANTHROPIC_API_KEY=\"sk-ant-...\"")
            self.client = anthropic.Anthropic()
        elif name == "openrouter":
            import requests  # noqa: F401

            self.key = os.environ.get("OPENROUTER_API_KEY")
            if not self.key:
                raise SystemExit("OPENROUTER_API_KEY is not set. In PowerShell: $env:OPENROUTER_API_KEY=\"sk-or-...\"")
        elif name != "mock":
            raise SystemExit(f"Unknown provider {name}")

    def ask(self, system: str, messages: list[dict]) -> tuple[str, dict]:
        """Returns (answer text, usage dict). Retries on transient errors."""
        for attempt in range(6):
            try:
                if self.name == "mock":
                    txt = messages[-1]["content"]
                    guess = "Geometrija" if re.search(r"troug|kru[zž]n|ugao|\\\\angle", txt, re.I) else "Algebra"
                    return f"Short reasoning.\nOblast: {guess}", {"input_tokens": len(system + txt) // 3, "output_tokens": 12}
                if self.name == "anthropic":
                    kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
                    if self.use_temperature:
                        kw["temperature"] = 0
                    try:
                        r = self.client.messages.create(**kw)
                    except Exception as e:  # some models do not accept temperature -> call without it
                        if self.use_temperature and "temperature" in str(e).lower():
                            self.use_temperature = False
                            self.temperature_note = "temperature not accepted by the model; default sampling"
                            continue
                        raise
                    txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    u = r.usage
                    return txt, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                                 "stop_reason": str(getattr(r, "stop_reason", "")),
                                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
                if self.name == "openrouter":
                    import requests

                    body = {"model": self.model, "max_tokens": self.max_tokens,
                            "messages": [{"role": "system", "content": system}] + messages}
                    if self.use_temperature:
                        body["temperature"] = 0
                    body.update(self.extra)
                    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=120,
                                      headers={"Authorization": f"Bearer {self.key}"}, json=body)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    j = r.json()
                    txt = j["choices"][0]["message"]["content"] or ""
                    u = j.get("usage", {})
                    return txt, {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                                 "stop_reason": "max_tokens" if j["choices"][0].get("finish_reason") == "length"
                                 else str(j["choices"][0].get("finish_reason", "")),
                                 "provider_model": j.get("model"), "provider_host": j.get("provider")}
            except SystemExit:
                raise
            except Exception as e:
                wait = min(60, 2 ** attempt * 2)
                print(f"   error ({e.__class__.__name__}: {str(e)[:120]}), retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("API not responding after 6 attempts")


class BatchClient:
    """Anthropic Message Batches API (or a fake version for testing, provider=mock)."""

    def __init__(self, prov: Provider, state_dir: Path):
        self.prov, self.state_dir = prov, state_dir
        if prov.name == "anthropic":  # newer SDK versions: client.messages.batches; older: client.beta...
            msgs = prov.client.messages
            self.api = msgs.batches if hasattr(msgs, "batches") else prov.client.beta.messages.batches

    def submit(self, reqs: list[dict]) -> str:
        if self.prov.name == "mock":
            bid = f"mockbatch_{int(time.time())}"
            (self.state_dir / f"{bid}.json").write_text(json.dumps(reqs, ensure_ascii=False), encoding="utf-8")
            return bid
        b = self.api.create(requests=reqs)
        return b.id

    def status(self, bid: str) -> tuple[str, dict]:
        if self.prov.name == "mock":
            return "ended", {"succeeded": "sve"}
        b = self.api.retrieve(bid)
        rc = b.request_counts
        return b.processing_status, {k: getattr(rc, k, None) for k in ("processing", "succeeded", "errored", "canceled", "expired")}

    def results(self, bid: str):
        """Yields (custom_id, text or None, usage dict)."""
        if self.prov.name == "mock":
            for r in json.loads((self.state_dir / f"{bid}.json").read_text(encoding="utf-8")):
                txt, u = self.prov.ask(r["params"]["system"], r["params"]["messages"])
                if r["custom_id"].endswith("3"):  # fake: every tenth request "fails" to test the fallback
                    yield r["custom_id"], None, {}
                else:
                    yield r["custom_id"], txt, u
            return
        for res in self.api.results(bid):
            if res.result.type == "succeeded":
                m = res.result.message
                txt = "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
                u = m.usage
                yield res.custom_id, txt, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                                           "stop_reason": str(getattr(m, "stop_reason", ""))}
            else:
                yield res.custom_id, None, {"batch_result": res.result.type}


def few_shot_messages(examples: pd.DataFrame) -> list[dict]:
    msgs = []
    for _, ex in examples.iterrows():
        msgs.append({"role": "user", "content": ex.model_text})
        msgs.append({"role": "assistant", "content": f"Oblast: {ex.oblast}"})
    return msgs


def flag(v) -> bool:
    """True only for a real True (pandas NaN for records without the column does not count)."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def group_bootstrap_ci(y, p, groups, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = [macro_f1(y[ii], p[ii]) for ii in (np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]) for _ in range(n))]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# --------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openrouter", "mock"])
    ap.add_argument("--model", default="mock")
    ap.add_argument("--shots", type=int, default=0, help="number of examples PER AREA (0 = zero-shot)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true", help="stratified random sample of --limit problems")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024, help="maximum answer length (including the model's reasoning)")
    ap.add_argument("--price-in", type=float, default=4.0, help="price per 1M input tokens (only for a usage estimate)")
    ap.add_argument("--price-out", type=float, default=20.0, help="price per 1M output tokens (only for a usage estimate)")
    ap.add_argument("--batch", choices=["submit", "status", "collect"], default=None,
                    help="Anthropic Message Batches API; without this option: standard calls")
    ap.add_argument("--or-host", default=None,
                    help="OpenRouter: pin one host (e.g. Alibaba), without fallback to other hosts")
    ap.add_argument("--extra-json", default=None,
                    help="extra parameters for the OpenRouter request as JSON (e.g. reasoning settings)")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check saved answers with the strict rule; send 1 follow-up question for those without an 'Oblast:' line")
    ap.add_argument("--max-retry", type=int, default=30,
                    help="batch collect: at most this many invalid answers are automatically retried with a standard call")
    args = ap.parse_args()

    df = pd.read_json(DATA, lines=True)
    folds = pd.read_csv(FOLDS)
    f0 = folds[folds.repeat == 0][["id", "fold"]]
    df = df.merge(f0, on="id")
    assert len(df) == df.id.nunique(), "every problem must appear exactly once in repetition 0"

    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", args.model)
    out = OUT_ROOT / f"llm_{args.provider}_{safe_model}_{args.shots}shot_{PROMPT_VERSION}"
    if args.or_host:
        out = out.with_name(out.name + f"_host-{args.or_host}")
    if args.pilot:
        out = out / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    work = df
    if args.pilot:
        n = args.limit or 50
        work = df.groupby("oblast").sample(frac=n / len(df), random_state=42)  # stratified by area
    elif args.limit:
        work = df.sample(args.limit, random_state=42)

    # few-shot examples per fold (only from the training part of that fold)
    shots = {}
    if args.shots > 0:
        for k in sorted(df.fold.unique()):
            train = df[df.fold != k]
            ex = train.groupby("oblast").sample(n=args.shots, random_state=42)
            shots[k] = ex.sample(frac=1, random_state=42)  # shuffled order of areas

    resp_path = out / "responses.jsonl"
    done = {}
    if resp_path.exists():
        for line in resp_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = work[~work.id.isin(done)]
    print(f"Model: {args.provider}/{args.model} | {args.shots}-shot | problems: {len(work)} | already done: {len(work) - len(todo)}")

    extra = json.loads(args.extra_json) if args.extra_json else {}
    if args.or_host:
        extra["provider"] = {"order": [args.or_host], "allow_fallbacks": False}
    prov = Provider(args.provider, args.model, args.max_tokens, extra)
    lock = threading.Lock()

    def build_msgs(row):
        return (few_shot_messages(shots[row.fold]) if args.shots else []) + [{"role": "user", "content": row.model_text}]

    def run_one(row, txt=None, usage=None, via_batch=False):
        msgs = build_msgs(row)
        if txt is None:  # standard call (or a batch request that failed)
            txt, usage = prov.ask(SYSTEM_PROMPT, msgs)
            via_batch = False
        lab = normalize_label(txt)
        retried = False
        if lab is None:
            retried = True
            msgs2 = msgs + [{"role": "assistant", "content": txt},
                            {"role": "user", "content": "Give only the final line: Oblast: <Algebra, Geometrija, Teorija brojeva or Kombinatorika>"}]
            txt2, usage2 = prov.ask(SYSTEM_PROMPT, msgs2)
            lab = normalize_label(txt2)
            usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2) if isinstance(usage.get(k, 0), (int, float))}
            txt = txt + " || " + txt2
        return {"id": row.id, "fold": int(row.fold), "true": row.oblast, "pred": lab or "INVALID", "raw": txt,
                "retried": retried, "usage": usage, "model": args.model, "provider": args.provider,
                "shots_per_class": args.shots, "prompt_version": PROMPT_VERSION, "batch": via_batch,
                "temperature": prov.temperature_note, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.revalidate:
        bad = [r for r in done.values() if r["id"] in set(work.id)
               and normalize_label(r["raw"].split(" || ")[-1]) is None]
        print(f"Answers without a valid final 'Oblast:' line: {len(bad)}")
        if len(bad) > args.max_retry:
            raise SystemExit(f"More than --max-retry {args.max_retry}; nothing was sent.")
        by_id = df.set_index("id")
        with resp_path.open("a", encoding="utf-8") as fout:
            for r in bad:
                row = by_id.loc[r["id"]].copy()
                row["id"] = r["id"]
                msgs = build_msgs(row) + [{"role": "assistant", "content": r["raw"].split(" || ")[0] or "(no answer)"},
                                          {"role": "user", "content": "Give only the final line: Oblast: <Algebra, Geometrija, Teorija brojeva or Kombinatorika>"}]
                txt2, u2 = prov.ask(SYSTEM_PROMPT, msgs)
                new = dict(r)
                new.update({"pred": normalize_label(txt2) or "INVALID", "raw": r["raw"] + " || " + txt2, "retried": True,
                            "revalidated": True, "previous_pred": r["pred"],
                            "usage": {k: r["usage"].get(k, 0) + u2.get(k, 0) for k in ("input_tokens", "output_tokens")}
                            | {"stop_reason": r["usage"].get("stop_reason", "")},
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                print(f"  {r['id']}: was {r['pred']} -> now {new['pred']} (true: {r['true']})")
                fout.write(json.dumps(new, ensure_ascii=False) + "\n")
                done[r["id"]] = new
        todo = work[~work.id.isin(done)]

    state = out / "batch_state.json"
    if args.batch:
        if args.provider == "openrouter":
            raise SystemExit("Batch mode is only available for --provider anthropic (or mock).")
        bc = BatchClient(prov, out)
        st = json.loads(state.read_text(encoding="utf-8")) if state.exists() else None
        if args.batch == "submit":
            if st and not st.get("collected"):
                raise SystemExit(f"Batch {st['batch_id']} was submitted but not collected yet. Run --batch collect.")
            if todo.empty:
                print("No unfinished problems - nothing to submit.")
                return 0
            id_map, reqs = {}, []
            for i, row in enumerate(todo.itertuples(index=False)):
                cid = f"q{i:05d}"
                id_map[cid] = row.id
                params = {"model": args.model, "max_tokens": args.max_tokens, "system": SYSTEM_PROMPT,
                          "messages": build_msgs(row)}
                reqs.append({"custom_id": cid, "params": params})
            bid = bc.submit(reqs)
            state.write_text(json.dumps({"batch_id": bid, "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "n_requests": len(reqs), "id_map": id_map, "collected": False},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Submitted batch {bid} with {len(reqs)} requests. Check with --batch status, collect with --batch collect.")
            return 0
        if not st:
            raise SystemExit("No submitted batch - run --batch submit first.")
        status, counts = bc.status(st["batch_id"])
        print(f"Batch {st['batch_id']} (submitted {st['submitted']}): status = {status}, {counts}")
        if args.batch == "status" or status != "ended":
            if args.batch == "collect":
                print("The batch has not finished yet - try again later.")
            return 0
        # collect: 1) download all answers; 2) failed requests are NOT re-sent (left for a new batch);
        #          3) invalid format: one standard follow-up request, only if there are at most --max-retry
        by_id = df.set_index("id")
        got, failed = [], []
        for cid, txt, usage in bc.results(st["batch_id"]):
            pid = st["id_map"][cid]
            if pid in done:
                continue
            (failed if txt is None else got).append((pid, txt, usage))
        invalid = [g for g in got if normalize_label(g[1]) is None]
        if len(invalid) > args.max_retry:
            print(f"WARNING: {len(invalid)} invalid answers (more than --max-retry {args.max_retry}). "
                  "Nothing was re-sent and nothing was written. Inspect some answers from the batch, "
                  "or rerun with a larger --max-retry to retry them.")
            return 0
        with resp_path.open("a", encoding="utf-8") as fout:
            for pid, txt, usage in got:
                row = by_id.loc[pid].copy()
                row["id"] = pid
                rec = run_one(row, txt, usage, via_batch=True)  # one follow-up call for an invalid format
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done[pid] = rec
        st["collected"] = True
        st["n_failed_in_batch_not_resent"] = len(failed)
        st["n_invalid_format_retried_standard"] = len(invalid)
        state.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Collected {len(got)} answers; invalid format retried with a standard call: {len(invalid)}.")
        if failed:
            print(f"{len(failed)} requests FAILED in the batch and were not re-sent. "
                  "Send them in a new batch: --batch submit, then --batch collect.")
            print("Metrics are computed only when every problem has an answer.")
            return 0
        todo = work[~work.id.isin(done)]

    n_done = 0
    with resp_path.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, r) for r in todo.itertuples(index=False)]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done[rec["id"]] = rec
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  {n_done}/{len(todo)}", flush=True)

    # ---- metrics ----
    res = pd.DataFrame([done[i] for i in work.id if i in done]).merge(df[["id", "group_id_v2"]], on="id")
    y, p = res["true"].to_numpy(), res["pred"].to_numpy()
    recs = res.to_dict("records")
    tok_in = sum(r["usage"].get("input_tokens", 0) for r in recs)
    tok_out = sum(r["usage"].get("output_tokens", 0) for r in recs)
    cost = sum((r["usage"].get("input_tokens", 0) * args.price_in + r["usage"].get("output_tokens", 0) * args.price_out)
               / 1e6 * (0.5 if flag(r.get("batch")) else 1.0) for r in recs)
    n_trunc = sum(1 for r in recs if r["usage"].get("stop_reason") == "max_tokens")
    fold_f1 = pd.Series({k: macro_f1(g["true"].to_numpy(), g["pred"].to_numpy()) for k, g in res.groupby("fold")})
    summary = {
        "provider": args.provider, "model": args.model, "shots_per_class": args.shots, "prompt_version": PROMPT_VERSION,
        "temperature": prov.temperature_note, "pilot": args.pilot, "n": int(len(res)),
        "macro_f1_pooled": round(macro_f1(y, p), 4),
        "macro_f1_fold_mean": round(float(fold_f1.mean()), 4), "macro_f1_fold_sd": round(float(fold_f1.std(ddof=1)), 4) if len(fold_f1) > 1 else None,
        "accuracy": round(accuracy_score(y, p), 4),
        "ci95_group_bootstrap": group_bootstrap_ci(y, p, res.group_id_v2.to_numpy()) if len(res) >= 50 else None,
        "n_invalid": int((p == "INVALID").sum()), "n_retried": int(res.retried.sum()),
        "per_class_f1": dict(zip(CLASSES, np.round(f1_score(y, p, labels=CLASSES, average=None, zero_division=0), 4).tolist())),
        "confusion": confusion_matrix(y, p, labels=CLASSES).tolist(),
        "tokens_in": int(tok_in), "tokens_out": int(tok_out),
        "n_truncated_max_tokens": int(n_trunc),
        "provider_hosts": pd.Series([r["usage"].get("provider_host") for r in recs]).dropna().value_counts().to_dict(),
        "n_revalidated": int(sum(1 for r in recs if flag(r.get("revalidated")))), "n_via_batch": int(sum(1 for r in recs if flag(r.get("batch")))),
        "estimated_cost_usd_this_run": round(cost, 2),
        "estimated_cost_usd_full_corpus": round(cost / max(len(res), 1) * len(df), 2),
    }
    # comparison with the linear models (same out-of-fold predictions of repetition 0), McNemar
    if OOF_LINEAR.exists():
        lin = pd.read_csv(OOF_LINEAR).set_index("id")
        cmp = {}
        for m in ("word_svm", "word_char_svm"):
            col = f"pred_{m}"
            if col in lin:
                q = lin.loc[res.id, col].to_numpy()
                a_ok, b_ok = q == y, p == y
                n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
                pv = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                cmp[m] = {"linear_macro_f1_same_items": round(macro_f1(y, q), 4),
                          "linear_only_correct": n01, "llm_only_correct": n10, "p_mcnemar": round(float(pv), 4)}
        summary["vs_linear_same_items"] = cmp
    res.drop(columns=["usage"]).to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["n_invalid"]:
        print(f"WARNING: {summary['n_invalid']} invalid answers - see column raw in predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
