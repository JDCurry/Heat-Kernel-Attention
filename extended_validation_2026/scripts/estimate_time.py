"""
Time Budget Planner for Diffusion-Native LM Experiments

Estimates training time based on your hardware throughput.
Run this to decide what's realistic before committing to a long run.
"""

import argparse


def estimate_time(
    throughput_steps_per_sec: float,
    dataset_size_tokens: int,
    batch_size: int,
    seq_len: int,
    n_epochs: int,
    n_runs: int,
):
    """Print time estimates for various training configurations."""

    tokens_per_step = batch_size * seq_len
    steps_per_epoch = dataset_size_tokens // tokens_per_step
    total_steps = steps_per_epoch * n_epochs

    print(f"\n{'='*70}")
    print(f"TIME BUDGET ESTIMATOR")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Throughput: {throughput_steps_per_sec:.2f} steps/sec")
    print(f"  Dataset: {dataset_size_tokens/1e6:.1f}M tokens")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq len: {seq_len}")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Total steps: {total_steps:,}")

    time_per_run_sec = total_steps / throughput_steps_per_sec
    time_per_run_hr = time_per_run_sec / 3600
    time_per_run_day = time_per_run_hr / 24

    print(f"\nPer run:")
    if time_per_run_day >= 1:
        print(f"  Time: {time_per_run_day:.1f} days ({time_per_run_hr:.1f} hours)")
    elif time_per_run_hr >= 1:
        print(f"  Time: {time_per_run_hr:.1f} hours")
    else:
        print(f"  Time: {time_per_run_sec/60:.1f} minutes")

    total_time_day = time_per_run_day * n_runs
    print(f"\nFor {n_runs} runs (full experiment program):")
    if total_time_day >= 1:
        print(f"  Total: {total_time_day:.1f} days")
    else:
        print(f"  Total: {total_time_day*24:.1f} hours")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")

    if total_time_day > 14:
        print(f"⚠️  {total_time_day:.0f} days is too long for 7 runs.")
        print(f"   Consider budget-limited runs:")
        print(f"   - 50K steps per run: ~{50000/throughput_steps_per_sec/3600:.1f} hr/run = "
              f"~{50000/throughput_steps_per_sec/3600 * n_runs / 24:.1f} days total")
        print(f"   - 100K steps per run: ~{100000/throughput_steps_per_sec/3600:.1f} hr/run = "
              f"~{100000/throughput_steps_per_sec/3600 * n_runs / 24:.1f} days total")
    elif total_time_day > 3:
        print(f"⚠️  {total_time_day:.0f} days is significant. Consider:")
        print(f"   - Running overnight in sequence")
        print(f"   - Reducing to 50-100K steps for initial iteration")
    else:
        print(f"✅  {total_time_day:.1f} days is very manageable.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--throughput", type=float, default=15.0,
                        help="Steps/sec (measure yourself; typical A4000 with AMP ~15-25)")
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=7,
                        help="Number of experimental runs (A1-A3, B1-B4 = 7)")

    args = parser.parse_args()

    dataset_sizes = {
        "wikitext-2": 2_000_000,       # ~2M BPE tokens
        "wikitext-103": 100_000_000,   # ~100M BPE tokens
    }

    estimate_time(
        throughput_steps_per_sec=args.throughput,
        dataset_size_tokens=dataset_sizes[args.dataset],
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_epochs=args.epochs,
        n_runs=args.runs,
    )

    print("\n\nKey comparisons:")
    print("-" * 70)

    # Show what different throughputs would mean
    print("\nImpact of throughput improvements:")
    for tp in [3.5, 7.0, 15.0, 25.0, 40.0]:
        total_steps = (dataset_sizes[args.dataset] // (args.batch_size * args.seq_len)) * args.epochs
        t_days = (total_steps / tp / 3600 / 24) * args.runs
        label = ""
        if abs(tp - 3.5) < 0.5:
            label = "  ← your current (bytes, no AMP)"
        elif abs(tp - 15.0) < 0.5:
            label = "  ← BPE + AMP (expected)"
        print(f"  {tp:5.1f} steps/s: {t_days:6.1f} days for {args.runs} runs{label}")


if __name__ == "__main__":
    main()
