"""Render a descriptive scoring diagnostic figure, not an accuracy benchmark.

python -m tieredkv.score_report PATH_TO_SUMMARY --out OUTPUT_DIRECTORY
"""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = data["rows"]
    if not data.get("diagnostic_only") or not rows:
        parser.error("expected a nonempty scoring diagnostic summary")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = sorted({r["layer"] for r in rows})
    grouped = {l: [r for r in rows if r["layer"] == l] for l in layers}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for key, label, color in [
        ("raw_position_spearman", "Raw accumulated score", "#475569"),
        ("mean_position_spearman", "Opportunity-normalized score", "#0f766e"),
    ]:
        points = [[r[key] for r in grouped[l] if r[key] is not None] for l in layers]
        means = [statistics.mean(x) if x else float("nan") for x in points]
        axes[0].plot(layers, means, label=label, color=color, linewidth=2)
    axes[0].axhline(0, color="#cbd5e1", linewidth=1)
    axes[0].set(xlabel="Model layer (zero-based)", ylabel="Mean Spearman correlation with token position",
                title="Normalization changes position association", ylim=(-1, 1))
    axes[0].legend(fontsize=8)
    overlap = [100*statistics.mean(r["top_overlap_fraction"] for r in grouped[l]
                                 if r["top_overlap_fraction"] is not None) for l in layers]
    axes[1].plot(layers, overlap, color="#b45309", linewidth=2)
    axes[1].set(xlabel="Model layer (zero-based)", ylabel="Raw/normalized top-group overlap (%)",
                title="Important-token selections differ", ylim=(0, 100))
    fig.suptitle(f"{data['model']} | {data['prompt_count']} diagnostic prompts", fontsize=12)
    fig.savefig(out / "scoring_diagnostics.png", dpi=180)
    fig.savefig(out / "scoring_diagnostics.svg")
    plt.close(fig)
    metrics = {
        "model": data["model"], "revision": data.get("revision"),
        "diagnostic_only": True, "prompt_count": data["prompt_count"],
        "layers": len(layers), "prompt_layer_records": len(rows),
        "raw_top_attention_mass_share_mean": statistics.mean(r["raw_top_attention_mass_share"] for r in rows),
        "mean_top_score_share_mean": statistics.mean(r["mean_top_score_share"] for r in rows),
        "top_overlap_fraction_mean": statistics.mean(r["top_overlap_fraction"] for r in rows if r["top_overlap_fraction"] is not None),
        "realized_top_fraction_range": [min(r["realized_top_fraction"] for r in rows), max(r["realized_top_fraction"] for r in rows)],
        "caveats": ["Short diagnostic prompts, not GSM8K/LongBench or the team's target model.",
                    "Means describe prompt-layer records, not 600 independent prompts.",
                    "Mean-normalized score share is not raw attention probability mass.",
                    "Position correlation or ranking differences do not prove better task accuracy.",
                    "ceil(5% * token_count) makes realized top fractions exceed 5% on some short prompts."]
    }
    (out / "diagnostic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
