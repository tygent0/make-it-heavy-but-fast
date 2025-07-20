import time
import statistics
from orchestrator import TaskOrchestrator

EXAMPLES = [
    # Elaborate prompts to better stress the orchestration logic
    "Research the long-term impact of artificial intelligence on renewable energy production, including potential breakthroughs, pitfalls, and cite at least five recent studies.",
    "Detail recent developments in global clean energy policy since 2022, focusing on regulatory incentives and subsidies.",
    "Compile the latest statistics on worldwide electric vehicle adoption and analyze regional trends and barriers to growth."
]


NUM_RUNS = 20


def benchmark():
    rows = []
    for prompt in EXAMPLES:
        tygent_times = []
        thread_times = []

        # Run with Tygent scheduler
        for _ in range(NUM_RUNS):
            orch_tygent = TaskOrchestrator(silent=True, use_tygent=True)
            start = time.perf_counter()
            try:
                orch_tygent.orchestrate(prompt)
            except Exception:
                pass
            tygent_times.append(time.perf_counter() - start)

        # Run with thread pool fallback
        for _ in range(NUM_RUNS):
            orch_thread = TaskOrchestrator(silent=True, use_tygent=False)
            start = time.perf_counter()
            try:
                orch_thread.orchestrate(prompt)
            except Exception:
                pass
            thread_times.append(time.perf_counter() - start)

        rows.append((prompt, tygent_times, thread_times))

    print(f"Benchmark Results over {NUM_RUNS} runs (seconds)")
    header = (
        f"{'Prompt':<40} {'Avg T':>10} {'Med T':>10} {'Std T':>10}"
        f" {'Avg Th':>10} {'Med Th':>10} {'Std Th':>10}"
    )
    print(header)
    for prompt, t_times, thr_times in rows:
        label = (prompt[:37] + '...') if len(prompt) > 40 else prompt
        t_mean = statistics.mean(t_times)
        t_median = statistics.median(t_times)
        t_stdev = statistics.stdev(t_times) if len(t_times) > 1 else 0.0
        thr_mean = statistics.mean(thr_times)
        thr_median = statistics.median(thr_times)
        thr_stdev = statistics.stdev(thr_times) if len(thr_times) > 1 else 0.0
        print(
            f"{label:<40} {t_mean:>10.2f} {t_median:>10.2f} {t_stdev:>10.2f}"
            f" {thr_mean:>10.2f} {thr_median:>10.2f} {thr_stdev:>10.2f}"
        )


if __name__ == '__main__':
    benchmark()
