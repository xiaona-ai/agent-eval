"""Benchmark runner for standard evaluation datasets.

Runs agent-eval-lite judges against FaithBench, HaluBench, etc.
Reports Accuracy, Precision, Recall, F1 with confidence intervals.
"""
import json, glob, os, sys, time, random, math
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent_eval import JudgeProvider, judge_faithfulness

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Metrics ──────────────────────────────────────────────

def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Compute Accuracy, Precision, Recall, F1 + 95% CI for accuracy."""
    n = len(y_true)
    if n == 0:
        return {}

    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # 95% CI for accuracy (Wilson interval)
    z = 1.96
    ci_low = (acc + z*z/(2*n) - z * math.sqrt((acc*(1-acc) + z*z/(4*n))/n)) / (1 + z*z/n)
    ci_high = (acc + z*z/(2*n) + z * math.sqrt((acc*(1-acc) + z*z/(4*n))/n)) / (1 + z*z/n)

    # Cohen's kappa
    pe = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp)) / (n*n)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0

    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(acc, 4),
        "accuracy_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "cohens_kappa": round(kappa, 4),
    }


# ── FaithBench Loader ────────────────────────────────────

def load_faithbench(max_samples: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load FaithBench samples with binary labels.

    Label logic: annotations with 'Unwanted' = hallucination (FAIL).
    Empty annotations or only 'Benign'/'Questionable' = PASS.
    """
    fb_dir = os.path.join(DATA_DIR, "FaithBench", "data_for_release")
    files = sorted(glob.glob(os.path.join(fb_dir, "batch_*.json")))
    if not files:
        raise FileNotFoundError(f"No FaithBench data in {fb_dir}")

    samples = []
    for f in files:
        batch = json.load(open(f))
        for s in batch["samples"]:
            # Determine if hallucinated: any 'Unwanted' annotation = FAIL
            has_unwanted = any(
                any("Unwanted" in l for l in a["label"])
                for a in s["annotations"]
            )
            samples.append({
                "id": f"fb_{os.path.basename(f)}_{s['sample_id']}",
                "context": s["source"],
                "output": s["summary"],
                "expected_faithful": not has_unwanted,
            })

    if max_samples and max_samples < len(samples):
        rng = random.Random(seed)
        # Stratified sampling: maintain label ratio
        pos = [s for s in samples if s["expected_faithful"]]
        neg = [s for s in samples if not s["expected_faithful"]]
        ratio = len(pos) / len(samples)
        n_pos = max(1, int(max_samples * ratio))
        n_neg = max_samples - n_pos
        rng.shuffle(pos)
        rng.shuffle(neg)
        samples = pos[:n_pos] + neg[:n_neg]
        rng.shuffle(samples)

    return samples


# ── HaluBench Loader (via HuggingFace API) ───────────────

def load_halubench(max_samples: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load HaluBench from HuggingFace datasets API.

    Format: passage + question → answer, label = PASS/FAIL
    """
    import urllib.request

    samples = []
    batch_size = 100
    # Fetch from multiple offsets to get balanced PASS/FAIL
    total_to_fetch = min((max_samples or 14900) * 3, 14900)  # over-fetch for balance
    offset = 0

    while len(samples) < total_to_fetch:
        url = (f"https://datasets-server.huggingface.co/rows?"
               f"dataset=PatronusAI%2FHaluBench&config=default&split=test"
               f"&offset={offset}&length={batch_size}")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  HaluBench fetch error at offset {offset}: {e}")
            break

        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r["row"]
            context = f"Passage: {row['passage']}\nQuestion: {row['question']}"
            samples.append({
                "id": row.get("id", f"hb_{offset}"),
                "context": context,
                "output": str(row["answer"]),
                "expected_faithful": row["label"] == "PASS",
            })

        offset += batch_size
        if len(rows) < batch_size:
            break

    if max_samples and max_samples < len(samples):
        rng = random.Random(seed)
        pos = [s for s in samples if s["expected_faithful"]]
        neg = [s for s in samples if not s["expected_faithful"]]
        ratio = len(pos) / len(samples)
        n_pos = max(1, int(max_samples * ratio))
        n_neg = max_samples - n_pos
        rng.shuffle(pos)
        rng.shuffle(neg)
        samples = pos[:n_pos] + neg[:n_neg]
        rng.shuffle(samples)

    return samples


# ── Benchmark Runner ─────────────────────────────────────

def run_faithfulness_benchmark(
    provider: JudgeProvider,
    samples: list[dict],
    dataset_name: str,
    verbose: bool = True,
    mode: str = "fast",
) -> dict:
    """Run faithfulness judge on a list of samples, compute metrics."""
    y_true = []
    y_pred = []
    errors = 0
    total_tokens = 0
    total_time = 0
    details = []

    print(f"\n{'─'*60}")
    print(f"  {dataset_name} | Model: {provider.model} | Samples: {len(samples)}")
    print(f"{'─'*60}")

    for i, sample in enumerate(samples):
        try:
            t = time.time()
            result = judge_faithfulness(
                provider,
                context=sample["context"],
                output=sample["output"],
                mode=mode,
            )
            elapsed = time.time() - t
            total_time += elapsed

            predicted_faithful = result.passed
            expected = sample["expected_faithful"]
            correct = predicted_faithful == expected

            y_true.append(expected)
            y_pred.append(predicted_faithful)

            tokens = result.judge_cost.total_tokens if result.judge_cost else 0
            total_tokens += tokens

            # RPM control: ~8 requests per minute max
            # thorough mode uses 2 API calls per sample
            delay = 15.0 if mode == "thorough" else 7.5
            time.sleep(delay)

            if verbose and not correct:
                print(f"  ✗ [{i+1}/{len(samples)}] id={sample['id']} "
                      f"expected={expected} got={predicted_faithful} ({tokens}tok, {elapsed:.1f}s)")
            elif verbose and (i + 1) % 20 == 0:
                acc_so_far = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
                print(f"  ● [{i+1}/{len(samples)}] running accuracy: {acc_so_far:.1%}")

            details.append({
                "id": sample["id"],
                "expected": expected,
                "predicted": predicted_faithful,
                "correct": correct,
                "tokens": tokens,
                "time_s": round(elapsed, 1),
            })

        except Exception as e:
            errors += 1
            y_true.append(sample["expected_faithful"])
            y_pred.append(None)
            if verbose:
                print(f"  ❌ [{i+1}/{len(samples)}] id={sample['id']} ERROR: {str(e)[:80]}")
            details.append({"id": sample["id"], "error": str(e)[:200]})

    # Remove None predictions for metrics
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if valid:
        vt, vp = zip(*valid)
        metrics = compute_metrics(list(vt), list(vp))
    else:
        metrics = {}

    metrics["errors"] = errors
    metrics["total_tokens"] = total_tokens
    metrics["total_time_s"] = round(total_time, 1)
    metrics["avg_time_s"] = round(total_time / max(1, len(samples) - errors), 1)

    print(f"{'─'*60}")
    print(f"  Results: Acc={metrics.get('accuracy','?')} "
          f"F1={metrics.get('f1','?')} "
          f"κ={metrics.get('cohens_kappa','?')} "
          f"| {total_tokens} tokens | {total_time:.0f}s")
    print(f"  CI95: {metrics.get('accuracy_ci95','?')}")
    print(f"  TP={metrics.get('tp',0)} TN={metrics.get('tn',0)} "
          f"FP={metrics.get('fp',0)} FN={metrics.get('fn',0)} Err={errors}")
    print(f"{'─'*60}")

    return {
        "dataset": dataset_name,
        "model": provider.model,
        "metrics": metrics,
        "details": details,
    }


# ── Main ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run standard benchmarks")
    parser.add_argument("--dataset", choices=["faithbench", "halubench", "all"], default="all")
    parser.add_argument("--model", default="grok-4.1-fast")
    parser.add_argument("--samples", type=int, default=100, help="Max samples per dataset")
    parser.add_argument("--api-key", default=os.environ.get("SORAI_API_KEY", ""))
    parser.add_argument("--base-url", default="https://newapi.sorai.me/v1")
    parser.add_argument("--mode", choices=["fast", "thorough"], default="fast",
                        help="Faithfulness mode: fast (single-call) or thorough (multi-step)")
    parser.add_argument("--output", default="benchmark_standard_results.json")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if not args.api_key:
        print("❌ Set SORAI_API_KEY or --api-key")
        sys.exit(1)

    provider = JudgeProvider(
        api_key=args.api_key, base_url=args.base_url,
        model=args.model, timeout=60,
    )

    all_results = []

    print("=" * 60)
    print(f"  agent-eval-lite — Standard Benchmark Suite")
    print(f"  Model: {args.model} | Samples/dataset: {args.samples} | Mode: {args.mode}")
    print("=" * 60)

    if args.dataset in ("faithbench", "all"):
        print("\nLoading FaithBench...")
        fb_samples = load_faithbench(max_samples=args.samples)
        pos = sum(1 for s in fb_samples if s["expected_faithful"])
        neg = len(fb_samples) - pos
        print(f"  Loaded {len(fb_samples)} samples (faithful={pos}, hallucinated={neg})")
        result = run_faithfulness_benchmark(provider, fb_samples, "FaithBench", args.verbose, mode=args.mode)
        all_results.append(result)

    if args.dataset in ("halubench", "all"):
        print("\nLoading HaluBench...")
        hb_samples = load_halubench(max_samples=args.samples)
        pos = sum(1 for s in hb_samples if s["expected_faithful"])
        neg = len(hb_samples) - pos
        print(f"  Loaded {len(hb_samples)} samples (faithful={pos}, hallucinated={neg})")
        result = run_faithfulness_benchmark(provider, hb_samples, "HaluBench", args.verbose, mode=args.mode)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['dataset']:15s} | Acc={m.get('accuracy','?'):>6} "
              f"F1={m.get('f1','?'):>6} κ={m.get('cohens_kappa','?'):>6} "
              f"| {m.get('total_tokens',0)} tok")
    print(f"{'='*60}\n")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


# ── JudgeBench Loader (via HuggingFace API) ─────────────

def load_judgebench(max_samples: Optional[int] = None, split: str = "gpt", seed: int = 42) -> list[dict]:
    """Load JudgeBench pairwise comparison samples.

    Args:
        max_samples: Max samples to load.
        split: "gpt" (350 samples) or "claude" (270 samples).
        seed: Random seed for sampling.

    Returns:
        List of dicts with prompt, response_a, response_b, expected_winner.
    """
    import urllib.request

    samples = []
    batch_size = 100
    offset = 0

    while True:
        url = (f"https://datasets-server.huggingface.co/rows?"
               f"dataset=ScalerLab%2FJudgeBench&config=default&split={split}"
               f"&offset={offset}&length={batch_size}")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  JudgeBench fetch error at offset {offset}: {e}")
            break

        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r["row"]
            label = row.get("label", "")
            # Label format: "A>B" or "B>A"
            if "A>B" in label:
                expected_a_wins = True
            elif "B>A" in label:
                expected_a_wins = False
            else:
                continue  # Skip ambiguous labels

            samples.append({
                "id": row.get("pair_id", f"jb_{offset}"),
                "source": row.get("source", ""),
                "prompt": row.get("question", ""),
                "response_a": row.get("response_A", ""),
                "response_b": row.get("response_B", ""),
                "expected_a_wins": expected_a_wins,
            })

        offset += batch_size
        if len(rows) < batch_size:
            break

    if max_samples and max_samples < len(samples):
        rng = random.Random(seed)
        # Stratified: maintain A>B vs B>A ratio
        a_wins = [s for s in samples if s["expected_a_wins"]]
        b_wins = [s for s in samples if not s["expected_a_wins"]]
        ratio = len(a_wins) / len(samples) if samples else 0.5
        n_a = max(1, int(max_samples * ratio))
        n_b = max_samples - n_a
        rng.shuffle(a_wins)
        rng.shuffle(b_wins)
        samples = a_wins[:n_a] + b_wins[:n_b]
        rng.shuffle(samples)

    return samples
