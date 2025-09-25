"""Command-line interface utilities for yolokernelgen."""

import argparse
from .knowledge_base import get_knowledge_base_stats, load_knowledge_base
from .storage import list_kernels


def show_stats(cache_dir: str = ".cache/yolokernelgen"):
    """Show knowledge base and cache statistics."""
    print("YoloKernelGen Statistics")
    print("=" * 40)

    # Knowledge base stats
    kb_stats = get_knowledge_base_stats(cache_dir)
    print(f"\nKnowledge Base:")
    print(f"  Successful kernels learned: {kb_stats['total_successful_kernels']}")
    print(f"  Operations types: {kb_stats['operations_learned']}")

    if kb_stats['operation_breakdown']:
        print("\n  Operation breakdown:")
        for op, stats in kb_stats['operation_breakdown'].items():
            print(f"    {op}: {stats['count']} kernels, avg {stats['avg_tokens']} tokens")

    # Cache stats
    all_kernels = list_kernels(cache_dir)
    correct_kernels = list_kernels(cache_dir, status_filter="correct")
    rejected_kernels = list_kernels(cache_dir, status_filter="rejected")

    print(f"\nCache Statistics:")
    print(f"  Total cached kernels: {len(all_kernels)}")
    print(f"  Successful (correct): {len(correct_kernels)}")
    print(f"  Failed (rejected): {len(rejected_kernels)}")

    if all_kernels:
        success_rate = len(correct_kernels) / len(all_kernels) * 100
        print(f"  Success rate: {success_rate:.1f}%")

    # Recent activity
    kb = load_knowledge_base(cache_dir)
    recent_successes = kb.get("successful_kernels", [])[-5:]
    if recent_successes:
        print(f"\nRecent successful kernels:")
        for kernel in recent_successes:
            operation = kernel.get("patterns", {}).get("operation", "unknown")
            score = kernel.get("patterns", {}).get("validation_score", 0)
            tokens = kernel.get("patterns", {}).get("tokens_used", 0)
            print(f"  • {operation}: {score}/10 tests passed, {tokens} tokens")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="YoloKernelGen CLI utilities")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--cache-dir", default=".cache/yolokernelgen", help="Cache directory")

    args = parser.parse_args()

    if args.stats:
        show_stats(args.cache_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()