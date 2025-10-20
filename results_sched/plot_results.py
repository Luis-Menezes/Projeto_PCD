import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_results(file_path):
    """Parse the test results file and extract timing data"""
    
    results = []
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find all test sections
    test_sections = re.findall(
        r'TESTANDO: (\d+) pontos, (\d+) clusters, Schedule: (\w+), Threads: (\d+) Chunk Sizes: (\d+).*?'
        r'Tempo: ([\d.]+) ms.*?'
        r'Tempo medido com omp_get_wtime\(\): ([\d.]+) segundos',
        content,
        re.DOTALL
    )
    
    for section in test_sections:
        n_points, k_clusters, schedule, threads, chunk_size, time_ms, time_seconds = section
        
        results.append({
            'n_points': int(n_points),
            'k_clusters': int(k_clusters),
            'schedule': schedule,
            'threads': int(threads),
            'chunk_size': int(chunk_size),
            'time_ms': float(time_ms),
            'time_seconds': float(time_seconds)
        })
    
    return results

def plot_schedule_comparison(results):
    """Create plots comparing different scheduling strategies"""
    
    # Group results by schedule type
    static_results = [r for r in results if r['schedule'] == 'static']
    dynamic_results = [r for r in results if r['schedule'] == 'dynamic']
    
    # Extract chunk sizes and times
    chunk_sizes = sorted(list(set([r['chunk_size'] for r in results])))
    
    static_times = []
    dynamic_times = []
    
    for chunk in chunk_sizes:
        static_time = next((r['time_seconds'] for r in static_results if r['chunk_size'] == chunk), None)
        dynamic_time = next((r['time_seconds'] for r in dynamic_results if r['chunk_size'] == chunk), None)
        
        static_times.append(static_time)
        dynamic_times.append(dynamic_time)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Line plot comparing schedules
    x_pos = np.arange(len(chunk_sizes))
    
    ax1.plot(x_pos, static_times, 'o-', label='Static', linewidth=2, markersize=8, color='blue')
    ax1.plot(x_pos, dynamic_times, 's-', label='Dynamic', linewidth=2, markersize=8, color='red')

    ax1.set_xlabel('Tamanho do Chunk', fontsize=12)
    ax1.set_ylabel('Tempo de execução (segundos)', fontsize=12)
    ax1.set_title('Comparação de Perfomance de Scheduling - OpenMP\n(1Mi pontos, 16 clusters, 32 threads)', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{c}' if c > 0 else 'default' for c in chunk_sizes])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (static_t, dynamic_t) in enumerate(zip(static_times, dynamic_times)):
        ax1.annotate(f'{static_t:.3f}s', (i, static_t), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='blue')
        ax1.annotate(f'{dynamic_t:.3f}s', (i, dynamic_t), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Plot 2: Bar chart comparison
    x = np.arange(len(chunk_sizes))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, static_times, width, label='Static', color='lightblue', edgecolor='blue')
    bars2 = ax2.bar(x + width/2, dynamic_times, width, label='Dynamic', color='lightcoral', edgecolor='red')
    
    ax2.set_xlabel('Tamanho do Chunk', fontsize=12)
    ax2.set_ylabel('Tempo de execução', fontsize=12)
    ax2.set_title('Performance do Schedule por Tamanho de Chunk\n(Bar Chart Comparison)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{c}' if c > 0 else 'default' for c in chunk_sizes])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_detailed_analysis(results):
    """Create detailed analysis plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    chunk_sizes = sorted(list(set([r['chunk_size'] for r in results])))
    static_results = [r for r in results if r['schedule'] == 'static']
    dynamic_results = [r for r in results if r['schedule'] == 'dynamic']
    
    # Plot 1: Relative performance (percentage difference from baseline)
    static_baseline = static_results[0]['time_seconds']  # chunk_size = 0
    dynamic_baseline = dynamic_results[0]['time_seconds']  # chunk_size = 0
    
    static_rel = [((r['time_seconds'] - static_baseline) / static_baseline * 100) 
                  for r in static_results]
    dynamic_rel = [((r['time_seconds'] - dynamic_baseline) / dynamic_baseline * 100) 
                   for r in dynamic_results]
    
    x_pos = np.arange(len(chunk_sizes))
    ax1.plot(x_pos, static_rel, 'o-', label='Static', linewidth=2, markersize=8, color='blue')
    ax1.plot(x_pos, dynamic_rel, 's-', label='Dynamic', linewidth=2, markersize=8, color='red')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Chunk Size')
    ax1.set_ylabel('Performance Change (%)')
    ax1.set_title('Relative Performance vs Default Chunk Size')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{c}' if c > 0 else 'default' for c in chunk_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup comparison
    fastest_time = min([r['time_seconds'] for r in results])
    static_speedup = [fastest_time / r['time_seconds'] for r in static_results]
    dynamic_speedup = [fastest_time / r['time_seconds'] for r in dynamic_results]
    
    ax2.plot(x_pos, static_speedup, 'o-', label='Static', linewidth=2, markersize=8, color='blue')
    ax2.plot(x_pos, dynamic_speedup, 's-', label='Dynamic', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Chunk Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup Relative to Best Performance')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{c}' if c > 0 else 'default' for c in chunk_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance variance
    static_times = [r['time_seconds'] for r in static_results]
    dynamic_times = [r['time_seconds'] for r in dynamic_results]
    
    variance_data = [static_times, dynamic_times]
    labels = ['Static', 'Dynamic']
    colors = ['lightblue', 'lightcoral']
    
    box_plot = ax3.boxplot(variance_data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Performance Distribution by Schedule Type')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary table as text
    ax4.axis('off')
    
    # Calculate statistics
    static_mean = np.mean(static_times)
    static_std = np.std(static_times)
    dynamic_mean = np.mean(dynamic_times)
    dynamic_std = np.std(dynamic_times)
    
    best_static = min(static_times)
    best_dynamic = min(dynamic_times)
    overall_best = min(best_static, best_dynamic)
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['time_seconds'])
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Dataset: {results[0]['n_points']:,} points, {results[0]['k_clusters']} clusters
    Threads: {results[0]['threads']}
    
    STATIC SCHEDULE:
    • Mean time: {static_mean:.3f}s ± {static_std:.3f}s
    • Best time: {best_static:.3f}s
    • Range: {max(static_times) - min(static_times):.3f}s
    
    DYNAMIC SCHEDULE:
    • Mean time: {dynamic_mean:.3f}s ± {dynamic_std:.3f}s
    • Best time: {best_dynamic:.3f}s
    • Range: {max(dynamic_times) - min(dynamic_times):.3f}s
    
    BEST CONFIGURATION:
    • Schedule: {best_result['schedule']}
    • Chunk size: {best_result['chunk_size'] if best_result['chunk_size'] > 0 else 'default'}
    • Time: {best_result['time_seconds']:.3f}s
    
    OBSERVATIONS:
    • Dynamic chunk=10 shows best performance
    • Static scheduling is more consistent
    • Chunk size impact is relatively small
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def main():
    # Parse results
    results_file = '/home/luis-menezes/Documents/Grad/2025_PCD/Projeto_PCD/results_sched/test_results_20251020_111846.txt'
    
    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found!")
        print("Please make sure the results file is in the same directory.")
        return
    
    print("Parsing results...")
    results = parse_results(results_file)
    
    if not results:
        print("No results found in the file!")
        return
    
    print(f"Found {len(results)} test results")
    
    # Create plots
    print("Creating comparison plots...")
    fig1 = plot_schedule_comparison(results)
    fig1.savefig('schedule_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Creating detailed analysis...")
    fig2 = plot_detailed_analysis(results)
    fig2.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    
    # Show plots
    plt.show()
    
    print("\nPlots saved as:")
    print("- schedule_comparison.png")
    print("- detailed_analysis.png")

if __name__ == "__main__":
    main()