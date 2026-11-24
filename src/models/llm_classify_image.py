"""VLM classification of Kangaroo problems directly from the IMAGE (zero-shot), on the same folds as OCR and CNN.

Providers:
  anthropic   - Claude (claude-opus-5-5); key: environment variable ANTHROPIC_API_KEY; use --batch
  local       - own server running vLLM (OpenAI-compatible API), e.g. Qwen/Qwen3.6-27B; --base-url
  openrouter  - the same through OpenRouter; environment variable OPENROUTER_API_KEY
  mock        - fake model to check that the script runs (no API calls)

Protocol (identical for all models, fixed after the pilot):
  - fixed prompt (PROMPT_VERSION), definitions of the 4 Kangaroo areas, the answer ends with 'Oblast: <area>';
  - zero-shot, one image per request, no OCR text;
  - --script lat or cyr: which version of the image is sent (each script is a separate run);
  - every answer is written to responses.jsonl immediately -> after an interruption the same command resumes;
  - invalid answer: one follow-up question, then INVALID (counted as an error).

Commands (from the repository root):
  dry run:
    python src/models/llm_classify_image.py --provider mock --script lat --limit 40
  Claude, pilot (50 images, standard calls; not used in the paper):
    python src/models/llm_classify_image.py --provider anthropic --model claude-opus-5-5 --script lat --pilot --limit 50
  Claude, whole corpus via the Message Batches API:
    python src/models/llm_classify_image.py --provider anthropic --model claude-opus-5-5 --script lat --batch submit
    python src/models/llm_classify_image.py --provider anthropic --model claude-opus-5-5 --script lat --batch status
    python src/models/llm_classify_image.py --provider anthropic --model claude-opus-5-5 --script lat --batch collect
  Qwen on a local vLLM server (e.g. through an SSH tunnel):
    python src/models/llm_classify_image.py --provider local --model Qwen/Qwen3.6-27B --base-url http://localhost:8000/v1 --script lat --pilot --limit 50
    python src/models/llm_classify_image.py --provider local --model Qwen/Qwen3.6-27B --base-url http://localhost:8000/v1 --script lat --workers 16

Output: reports/image/v2/vlm_<provider>_<model>_<script>_<PROMPT_VERSION>[/pilot]/
    responses.jsonl, predictions.csv, summary.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
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
ANNOT = REPO_ROOT / "reports" / "image" / "final_image_annotations.csv"
FOLDS = REPO_ROOT / "reports" / "image" / "v2" / "image_folds_v2.csv"
OCR_OOF = REPO_ROOT / "reports" / "image" / "v2" / "ocr_text_cv" / "oof_predictions.csv"
OUT_ROOT = REPO_ROOT / "reports" / "image" / "v2"
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Logika"]
MAX_SIDE = 1568  # longer image side; larger images are downscaled (same for all models)

PROMPT_VERSION = "kangaroo-img-v1"
SYSTEM_PROMPT = """You classify problems from the Kangaroo mathematics competition (Serbia) by mathematical area.
You receive an image of one problem. It is in Serbian (Cyrillic or Latin script) and may contain a figure
and answer options. The competition covers all school ages, from grade 1 to grade 12.
Choose exactly ONE of the following four areas:
- Algebra: calculations and arithmetic, expressions, equations and inequalities, word problems solved by computing quantities (ages, prices, ratios, percentages, speed, time), functions, sequences.
- Geometrija: shapes and figures, lengths, perimeters, angles, areas, volumes, solids and nets, symmetry, geometric transformations, coordinates.
- Teorija brojeva: properties of whole numbers: digits, divisibility, even and odd, primes, remainders, multiples and divisors, number representations.
- Logika: logical reasoning, true and false statements, ordering and arrangements, counting cases and combinatorics, patterns, games and strategies, puzzles.
If a problem touches several areas, choose the one that is central to solving it.
You may think briefly first. Your answer must end with one final line of exactly this form:
Oblast: <area>
where <area> is one of: Algebra, Geometrija, Teorija brojeva, Logika."""
USER_TEXT = "Classify this problem."
FOLLOW_UP = "Give only the final line: Oblast: <Algebra, Geometrija, Teorija brojeva or Logika>"


# --------------------------------------------------------------------------------------------
def normalize_label(text: str) -> str | None:
    t0 = unicodedata.normalize("NFC", text or "")
    m = re.findall(r"oblast\s*[:\-]\s*\**\s*([^\n*]+)", t0, flags=re.I)
    if m:
        t0 = m[-1]
    t = re.sub(r"\s+", " ", t0.strip().strip(" .:*\"'`\n").lower())
    for c in CLASSES:
        if t == c.lower():
            return c
    if m:
        hits = [c for c in CLASSES if c.lower() in t]
        return hits[0] if len(hits) == 1 else None
    return None


def encode_image(path: Path) -> tuple[str, str]:
    """PNG as base64 (downscaled only if the longer side > MAX_SIDE)."""
    from PIL import Image

    im = Image.open(path)
    im.load()
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "image/png", base64.b64encode(buf.getvalue()).decode("ascii")


def flag(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and bool(v)


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def group_bootstrap_ci(y, p, groups, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = [macro_f1(y[ii], p[ii]) for ii in (np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]) for _ in range(n))]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"only_a_correct": b, "only_b_correct": c, "p": round(p, 4)}


# --------------------------------------------------------------------------------------------
class Provider:
    def __init__(self, name, model, max_tokens, base_url=None, extra=None, temperature=0.0, timeout=900):
        self.name, self.model, self.max_tokens = name, model, max_tokens
        self.timeout = timeout
        self.base_url, self.extra = base_url, extra or {}
        self.temperature = temperature
        self.use_temperature = True
        self.temperature_note = f"temperature={temperature}"
        if name == "anthropic":
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY is not set. In PowerShell: $env:ANTHROPIC_API_KEY=\"sk-ant-...\"")
            self.client = anthropic.Anthropic()
        elif name == "openrouter":
            self.key = os.environ.get("OPENROUTER_API_KEY")
            if not self.key:
                raise SystemExit("OPENROUTER_API_KEY is not set. In PowerShell: $env:OPENROUTER_API_KEY=\"sk-or-...\"")
            self.base_url = "https://openrouter.ai/api/v1"
        elif name == "local":
            if not base_url:
                raise SystemExit("--provider local requires --base-url (e.g. http://localhost:8000/v1)")
            self.key = os.environ.get("LOCAL_API_KEY", "EMPTY")
        elif name != "mock":
            raise SystemExit(f"Unknown provider {name}")

    # messages in the format of each API
    def user_msg(self, media, b64):
        if self.name == "anthropic":
            return {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": USER_TEXT}]}
        if self.name == "mock":
            return {"role": "user", "content": USER_TEXT}
        return {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
            {"type": "text", "text": USER_TEXT}]}

    def ask(self, messages):
        for attempt in range(6):
            try:
                if self.name == "mock":
                    import random

                    return f"Short.\nOblast: {random.Random(len(str(messages))).choice(CLASSES)}", \
                        {"input_tokens": 900, "output_tokens": 150, "stop_reason": "end_turn"}
                if self.name == "anthropic":
                    kw = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT, messages=messages)
                    if self.use_temperature:
                        kw["temperature"] = self.temperature
                    try:
                        r = self.client.messages.create(**kw)
                    except Exception as e:
                        if self.use_temperature and "temperature" in str(e).lower():
                            self.use_temperature = False
                            self.temperature_note = "temperature not accepted by the model; default sampling"
                            continue
                        raise
                    txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    return txt, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                                 "stop_reason": str(getattr(r, "stop_reason", ""))}
                import requests

                body = {"model": self.model, "max_tokens": self.max_tokens,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        "temperature": self.temperature}
                body.update(self.extra)
                r = requests.post(f"{self.base_url.rstrip('/')}/chat/completions", timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"}, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise SystemExit(f"Error {r.status_code}: {r.text[:500]}")
                j = r.json()
                ch = j["choices"][0]
                txt = ch["message"].get("content") or ""
                u = j.get("usage", {})
                return txt, {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                             "stop_reason": "max_tokens" if ch.get("finish_reason") == "length" else str(ch.get("finish_reason", "")),
                             "provider_host": j.get("provider")}
            except SystemExit:
                raise
            except Exception as e:
                wait = min(60, 2 ** attempt * 2)
                print(f"   error ({e.__class__.__name__}: {str(e)[:120]}), retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("API not responding after 6 attempts")


# --------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "local", "openrouter", "mock"])
    ap.add_argument("--model", default="mock")
    ap.add_argument("--script", required=True, choices=["lat", "cyr"], help="which image version: Latin or Cyrillic script")
    ap.add_argument("--base-url", default=None, help="for --provider local, e.g. http://localhost:8000/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: 1024 for Claude (as for text); 32768 for Qwen (reasoning without a practical limit, as for text)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for one answer (long reasoning: 7200)")
    ap.add_argument("--price-in", type=float, default=None, help="price per 1M input tokens (only for a usage estimate)")
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--batch", choices=["submit", "status", "collect"], default=None)
    ap.add_argument("--batch-size", type=int, default=1000, help="requests per batch (batches have a size limit)")
    ap.add_argument("--max-retry", type=int, default=30)
    ap.add_argument("--extra-json", default=None)
    ap.add_argument("--or-host", default=None)
    args = ap.parse_args()

    is_claude = args.provider == "anthropic"
    if args.max_tokens is None:
        args.max_tokens = 1024 if is_claude or args.provider == "mock" else 32768
    if args.price_in is None:
        args.price_in, args.price_out = {"anthropic": (4.0, 20.0), "openrouter": (0.30, 2.0)}.get(args.provider, (0.0, 0.0))

    ann = pd.read_csv(ANNOT, dtype={"task_family_id": str})
    folds = pd.read_csv(FOLDS, dtype=str)
    col = "latin_image_path" if args.script == "lat" else "cyrillic_image_path"
    df = folds.merge(ann[["task_family_id", col]], on="task_family_id", how="left")
    if df[col].isna().any():
        raise SystemExit(f"{df[col].isna().sum()} families have no image path")
    df["fold"] = df.fold.astype(int)
    df["path"] = df[col].map(lambda p: REPO_ROOT / p)
    missing = [p for p in df.path if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} images, e.g. {missing[0]}")

    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", args.model)
    out = OUT_ROOT / f"vlm_{args.provider}_{safe_model}_{args.script}_{PROMPT_VERSION}"
    if args.or_host:
        out = out.with_name(out.name + f"_host-{args.or_host}")
    if args.pilot:
        out = out / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    work = df
    if args.pilot:
        n = args.limit or 50
        work = df.groupby("label").sample(frac=n / len(df), random_state=42)
    elif args.limit:
        work = df.sample(args.limit, random_state=42)

    resp_path = out / "responses.jsonl"
    done = {}
    if resp_path.exists():
        for line in resp_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = work[~work.task_family_id.isin(done)]
    print(f"Model: {args.provider}/{args.model} | script: {args.script} | images: {len(work)} | already done: {len(work) - len(todo)}")

    extra = json.loads(args.extra_json) if args.extra_json else {}
    if args.or_host:
        extra["provider"] = {"order": [args.or_host], "allow_fallbacks": False}
    prov = Provider(args.provider, args.model, args.max_tokens, args.base_url, extra, args.temperature, args.timeout)
    lock = threading.Lock()

    def record(row, txt, usage, via_batch):
        msgs = [prov.user_msg(*encode_image(row.path))]
        lab = normalize_label(txt)
        retried = False
        if lab is None:
            retried = True
            txt2, u2 = prov.ask(msgs + [{"role": "assistant", "content": txt or "(no answer)"},
                                        {"role": "user", "content": FOLLOW_UP}])
            lab = normalize_label(txt2)
            usage = {"input_tokens": usage.get("input_tokens", 0) + u2.get("input_tokens", 0),
                     "output_tokens": usage.get("output_tokens", 0) + u2.get("output_tokens", 0),
                     "stop_reason": usage.get("stop_reason", "")}
            txt = (txt or "") + " || " + txt2
        return {"id": row.task_family_id, "fold": int(row.fold), "true": row.label, "pred": lab or "INVALID",
                "raw": txt, "retried": retried, "usage": usage, "model": args.model, "provider": args.provider,
                "script": args.script, "prompt_version": PROMPT_VERSION, "batch": via_batch,
                "max_tokens": args.max_tokens,
                "temperature": "default sampling (Batch API, temperature not set)" if via_batch else prov.temperature_note,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    def run_one(row):
        txt, usage = prov.ask([prov.user_msg(*encode_image(row.path))])
        return record(row, txt, usage, False)

    # ---------------- Batch (Claude only) ----------------
    state = out / "batch_state.json"
    if args.batch:
        if args.provider not in ("anthropic", "mock"):
            raise SystemExit("Batch mode is only available for --provider anthropic.")
        st = json.loads(state.read_text(encoding="utf-8")) if state.exists() else None
        if args.provider == "anthropic":
            msgs_api = prov.client.messages
            api = msgs_api.batches if hasattr(msgs_api, "batches") else prov.client.beta.messages.batches
        if args.batch == "submit":
            if st and not st.get("collected"):
                raise SystemExit("Submitted batches have not been collected yet. Run --batch collect.")
            if todo.empty:
                print("No unfinished images - nothing to submit.")
                return 0
            id_map, batches = {}, []
            rows = list(todo.itertuples(index=False))
            for start in range(0, len(rows), args.batch_size):
                reqs = []
                for i, row in enumerate(rows[start:start + args.batch_size], start):
                    cid = f"q{i:05d}"
                    id_map[cid] = row.task_family_id
                    reqs.append({"custom_id": cid, "params": {
                        "model": args.model, "max_tokens": args.max_tokens, "system": SYSTEM_PROMPT,
                        "messages": [prov.user_msg(*encode_image(row.path))]}})
                if args.provider == "mock":
                    bid = f"mock_{start}"
                else:
                    bid = api.create(requests=reqs).id
                batches.append(bid)
                print(f"  submitted batch {bid}: {len(reqs)} requests")
            state.write_text(json.dumps({"batch_ids": batches, "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "n_requests": len(rows), "id_map": id_map, "collected": False},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Submitted {len(rows)} requests in {len(batches)} batches. Check with --batch status, collect with --batch collect.")
            return 0
        if not st:
            raise SystemExit("No submitted batches - run --batch submit first.")
        if st.get("collected"):
            print("The batches have already been collected.")
        else:
            all_ended = True
            for bid in st["batch_ids"]:
                if args.provider == "mock":
                    print(f"  {bid}: ended")
                    continue
                b = api.retrieve(bid)
                rc = b.request_counts
                print(f"  {bid}: {b.processing_status}, succeeded {rc.succeeded}, processing {rc.processing}, errored {rc.errored}, expired {rc.expired}")
                all_ended &= b.processing_status == "ended"
            if args.batch == "status" or not all_ended:
                if args.batch == "collect":
                    print("Not all batches have finished - try again later.")
                return 0
            by_id = df.set_index("task_family_id")
            got, failed = [], []
            for bid in st["batch_ids"]:
                if args.provider == "mock":
                    for cid, pid in st["id_map"].items():
                        if int(cid[1:]) // args.batch_size * args.batch_size == int(bid.split("_")[1]):
                            got.append((pid, f"x\nOblast: {CLASSES[int(cid[1:]) % 4]}", {"input_tokens": 900, "output_tokens": 150}))
                    continue
                for res in api.results(bid):
                    pid = st["id_map"][res.custom_id]
                    if res.result.type == "succeeded":
                        m = res.result.message
                        txt = "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
                        got.append((pid, txt, {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens,
                                               "stop_reason": str(getattr(m, "stop_reason", ""))}))
                    else:
                        failed.append(pid)
            got = [g for g in got if g[0] not in done]
            invalid = [g for g in got if normalize_label(g[1]) is None]
            if len(invalid) > args.max_retry:
                print(f"WARNING: {len(invalid)} invalid answers (more than --max-retry {args.max_retry}). "
                      "Nothing was re-sent and nothing was written. Inspect the answers before continuing.")
                return 0
            with resp_path.open("a", encoding="utf-8") as fout:
                for pid, txt, usage in got:
                    row = by_id.loc[pid].copy()
                    row["task_family_id"] = pid
                    rec = record(row, txt, usage, True)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[pid] = rec
            st.update({"collected": True, "n_failed_not_resent": len(failed), "n_invalid_retried": len(invalid)})
            state.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Collected {len(got)} answers; invalid format retried with one follow-up question: {len(invalid)}.")
            if failed:
                print(f"{len(failed)} requests failed in the batch. Send them in a new batch: --batch submit, then --batch collect.")
                return 0
        todo = work[~work.task_family_id.isin(done)]
        if not todo.empty:
            print(f"Still {len(todo)} images without an answer - submit them: --batch submit.")
            return 0

    # ---------------- standard calls ----------------
    n_done = 0
    if not todo.empty:
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

    # ---------------- metrics ----------------
    res = pd.DataFrame([done[i] for i in work.task_family_id if i in done]).merge(
        df[["task_family_id", "group_id_v2", "category"]].rename(columns={"task_family_id": "id"}), on="id")
    y, p = res["true"].to_numpy(), res["pred"].to_numpy()
    recs = res.to_dict("records")
    tok_in = sum(r["usage"].get("input_tokens", 0) for r in recs)
    tok_out = sum(r["usage"].get("output_tokens", 0) for r in recs)
    cost = sum((r["usage"].get("input_tokens", 0) * args.price_in + r["usage"].get("output_tokens", 0) * args.price_out)
               / 1e6 * (0.5 if flag(r.get("batch")) else 1.0) for r in recs)
    fold_f1 = pd.Series({k: macro_f1(g["true"].to_numpy(), g["pred"].to_numpy()) for k, g in res.groupby("fold")})
    summary = {
        "provider": args.provider, "model": args.model, "script": args.script, "prompt_version": PROMPT_VERSION,
        "temperature": sorted({str(r.get("temperature")) for r in recs}), "max_tokens": args.max_tokens, "pilot": args.pilot, "n": int(len(res)),
        "macro_f1_pooled": round(macro_f1(y, p), 4),
        "macro_f1_fold_mean": round(float(fold_f1.mean()), 4),
        "macro_f1_fold_sd": round(float(fold_f1.std(ddof=1)), 4) if len(fold_f1) > 1 else None,
        "accuracy": round(accuracy_score(y, p), 4),
        "ci95_group_bootstrap": group_bootstrap_ci(y, p, res.group_id_v2.to_numpy()) if len(res) >= 50 else None,
        "n_invalid": int((p == "INVALID").sum()), "n_retried": int(res.retried.sum()),
        "n_truncated_max_tokens": int(sum(1 for r in recs if r["usage"].get("stop_reason") == "max_tokens")),
        "per_class_f1": dict(zip(CLASSES, np.round(f1_score(y, p, labels=CLASSES, average=None, zero_division=0), 4).tolist())),
        "per_category_macro_f1": {k: round(macro_f1(g["true"].to_numpy(), g["pred"].to_numpy()), 4)
                                  for k, g in res.groupby("category")},
        "confusion": confusion_matrix(y, p, labels=CLASSES).tolist(),
        "tokens_in": int(tok_in), "tokens_out": int(tok_out),
        "mean_output_tokens": round(tok_out / max(len(res), 1), 1),
        "n_via_batch": int(sum(1 for r in recs if flag(r.get("batch")))),
        "estimated_cost_usd_this_run": round(cost, 2),
        "estimated_cost_usd_full_corpus": round(cost / max(len(res), 1) * len(df), 2),
    }
    if OCR_OOF.exists() and len(res) == len(df):
        ocr = pd.read_csv(OCR_OOF, dtype={"task_family_id": str}).set_index("task_family_id")
        ocr_col = f"pred_ocr_{args.script}"
        o = ocr.loc[res.id, ocr_col].to_numpy()
        summary["mcnemar_vs_ocr_same_script"] = mcnemar(p == y, o == y)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    res.drop(columns=["usage"]).to_csv(out / "predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
