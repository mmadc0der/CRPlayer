"""
Dataset inspection and analysis utilities for screen page classification.
Provides comprehensive analysis of dataset characteristics, class balance, and data quality.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from data_loader import DatasetInspector, DatasetConfig

logger = logging.getLogger(__name__)
console = Console()


class DatasetAnalyzer:
    """Comprehensive dataset analysis and visualization."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.inspector = DatasetInspector(config)
    
    def generate_comprehensive_report(self, dataset_id: int, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive analysis report for a dataset."""
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Generating comprehensive report for dataset {dataset_id}[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Step 1: Download dataset
            task1 = progress.add_task("Downloading dataset...", total=None)
            download_result = self.inspector.download_dataset(dataset_id, output_dir)
            progress.update(task1, description="Dataset downloaded")
            
            # Step 2: Analyze class balance
            task2 = progress.add_task("Analyzing class balance...", total=None)
            balance_analysis = self.inspector.analyze_class_balance(dataset_id)
            progress.update(task2, description="Class balance analyzed")
            
            # Step 3: Load data for detailed analysis
            task3 = progress.add_task("Loading data for detailed analysis...", total=None)
            labeled_data = download_result['dataframe']
            progress.update(task3, description="Data loaded")
            
            # Step 4: Generate visualizations
            task4 = progress.add_task("Generating visualizations...", total=None)
            self._generate_visualizations(labeled_data, balance_analysis, output_path)
            progress.update(task4, description="Visualizations generated")
            
            # Step 5: Generate summary statistics
            task5 = progress.add_task("Generating summary statistics...", total=None)
            summary_stats = self._generate_summary_statistics(labeled_data, balance_analysis)
            progress.update(task5, description="Summary statistics generated")
        
        # Save report
        report = {
            'dataset_id': dataset_id,
            'download_result': download_result,
            'balance_analysis': balance_analysis,
            'summary_statistics': summary_stats,
            'output_directory': str(output_path)
        }
        
        with open(output_path / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display summary
        self._display_summary_report(report)
        
        console.print(f"[green]Analysis complete! Report saved to: {output_path}[/green]")
        
        return report
    
    def _generate_visualizations(self, df: pd.DataFrame, balance_analysis: Dict[str, Any], output_path: Path):
        """Generate comprehensive visualizations."""
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # 1. Class distribution bar plot
        plt.figure(figsize=fig_size)
        class_counts = df['class_name'].value_counts()
        bars = plt.bar(range(len(class_counts)), class_counts.values)
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Class balance pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Class Balance (Percentage)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_path / 'class_balance_pie.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Samples per session
        plt.figure(figsize=fig_size)
        session_counts = df['session_id'].value_counts()
        plt.bar(range(len(session_counts)), session_counts.values)
        plt.xlabel('Session')
        plt.ylabel('Number of Samples')
        plt.title('Samples per Session')
        plt.xticks(range(len(session_counts)), session_counts.index, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'samples_per_session.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Class distribution per session (heatmap)
        if len(df['session_id'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            pivot_table = df.groupby(['session_id', 'class_name']).size().unstack(fill_value=0)
            sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues')
            plt.title('Class Distribution per Session')
            plt.xlabel('Class')
            plt.ylabel('Session')
            plt.tight_layout()
            plt.savefig(output_path / 'class_distribution_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Imbalance ratio visualization
        plt.figure(figsize=(10, 6))
        imbalance_ratio = balance_analysis.get('imbalance_ratio', 0)
        plt.bar(['Imbalance Ratio'], [imbalance_ratio], color='red' if imbalance_ratio > 10 else 'orange' if imbalance_ratio > 5 else 'green')
        plt.ylabel('Ratio (Most Common / Least Common)')
        plt.title(f'Dataset Imbalance Ratio: {imbalance_ratio:.2f}')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
        plt.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Moderate Imbalance')
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='High Imbalance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'imbalance_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_statistics(self, df: pd.DataFrame, balance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        
        # Basic statistics
        total_samples = len(df)
        num_classes = df['class_name'].nunique()
        num_sessions = df['session_id'].nunique()
        
        # Class statistics
        class_counts = df['class_name'].value_counts()
        class_stats = {
            'most_common': class_counts.index[0],
            'most_common_count': int(class_counts.iloc[0]),
            'least_common': class_counts.index[-1],
            'least_common_count': int(class_counts.iloc[-1]),
            'mean_samples_per_class': float(class_counts.mean()),
            'std_samples_per_class': float(class_counts.std()),
            'median_samples_per_class': float(class_counts.median())
        }
        
        # Session statistics
        session_counts = df['session_id'].value_counts()
        session_stats = {
            'samples_per_session_mean': float(session_counts.mean()),
            'samples_per_session_std': float(session_counts.std()),
            'samples_per_session_median': float(session_counts.median()),
            'sessions_with_most_samples': session_counts.index[0],
            'sessions_with_least_samples': session_counts.index[-1]
        }
        
        # Imbalance metrics
        imbalance_ratio = balance_analysis.get('imbalance_ratio', float('inf'))
        gini_coefficient = self._calculate_gini_coefficient(class_counts.values)
        
        # Data quality metrics
        missing_paths = df['frame_path_rel'].isna().sum()
        duplicate_samples = df.duplicated(subset=['session_id', 'frame_id']).sum()
        
        return {
            'basic_stats': {
                'total_samples': total_samples,
                'num_classes': num_classes,
                'num_sessions': num_sessions
            },
            'class_statistics': class_stats,
            'session_statistics': session_stats,
            'imbalance_metrics': {
                'imbalance_ratio': imbalance_ratio,
                'gini_coefficient': gini_coefficient,
                'balance_quality': self._assess_balance_quality(imbalance_ratio, gini_coefficient)
            },
            'data_quality': {
                'missing_paths': int(missing_paths),
                'duplicate_samples': int(duplicate_samples),
                'data_quality_score': self._calculate_data_quality_score(missing_paths, duplicate_samples, total_samples)
            }
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    def _assess_balance_quality(self, imbalance_ratio: float, gini_coefficient: float) -> str:
        """Assess the quality of class balance."""
        if imbalance_ratio <= 2 and gini_coefficient <= 0.2:
            return "Excellent"
        elif imbalance_ratio <= 5 and gini_coefficient <= 0.4:
            return "Good"
        elif imbalance_ratio <= 10 and gini_coefficient <= 0.6:
            return "Moderate"
        else:
            return "Poor"
    
    def _calculate_data_quality_score(self, missing_paths: int, duplicate_samples: int, total_samples: int) -> float:
        """Calculate overall data quality score (0-100)."""
        missing_penalty = (missing_paths / total_samples) * 50 if total_samples > 0 else 0
        duplicate_penalty = (duplicate_samples / total_samples) * 30 if total_samples > 0 else 0
        return max(0, 100 - missing_penalty - duplicate_penalty)
    
    def _display_summary_report(self, report: Dict[str, Any]):
        """Display a formatted summary report in the console."""
        
        # Basic statistics table
        basic_stats = report['summary_statistics']['basic_stats']
        table1 = Table(title="Dataset Overview")
        table1.add_column("Metric", style="cyan")
        table1.add_column("Value", style="magenta")
        
        table1.add_row("Total Samples", str(basic_stats['total_samples']))
        table1.add_row("Number of Classes", str(basic_stats['num_classes']))
        table1.add_row("Number of Sessions", str(basic_stats['num_sessions']))
        
        console.print(table1)
        
        # Class balance table
        class_stats = report['summary_statistics']['class_statistics']
        table2 = Table(title="Class Balance")
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="magenta")
        
        table2.add_row("Most Common Class", f"{class_stats['most_common']} ({class_stats['most_common_count']} samples)")
        table2.add_row("Least Common Class", f"{class_stats['least_common']} ({class_stats['least_common_count']} samples)")
        table2.add_row("Mean Samples per Class", f"{class_stats['mean_samples_per_class']:.1f}")
        table2.add_row("Std Samples per Class", f"{class_stats['std_samples_per_class']:.1f}")
        
        console.print(table2)
        
        # Imbalance metrics
        imbalance_metrics = report['summary_statistics']['imbalance_metrics']
        table3 = Table(title="Imbalance Assessment")
        table3.add_column("Metric", style="cyan")
        table3.add_column("Value", style="magenta")
        
        table3.add_row("Imbalance Ratio", f"{imbalance_metrics['imbalance_ratio']:.2f}")
        table3.add_row("Gini Coefficient", f"{imbalance_metrics['gini_coefficient']:.3f}")
        table3.add_row("Balance Quality", imbalance_metrics['balance_quality'])
        
        console.print(table3)
        
        # Data quality
        data_quality = report['summary_statistics']['data_quality']
        table4 = Table(title="Data Quality")
        table4.add_column("Metric", style="cyan")
        table4.add_column("Value", style="magenta")
        
        table4.add_row("Missing Paths", str(data_quality['missing_paths']))
        table4.add_row("Duplicate Samples", str(data_quality['duplicate_samples']))
        table4.add_row("Quality Score", f"{data_quality['data_quality_score']:.1f}/100")
        
        console.print(table4)
        
        # Recommendations
        recommendations = self._generate_recommendations(report)
        if recommendations:
            console.print(Panel("\n".join(recommendations), title="Recommendations", border_style="yellow"))
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        imbalance_ratio = report['summary_statistics']['imbalance_metrics']['imbalance_ratio']
        balance_quality = report['summary_statistics']['imbalance_metrics']['balance_quality']
        data_quality_score = report['summary_statistics']['data_quality']['data_quality_score']
        
        if balance_quality == "Poor":
            recommendations.append("⚠️  High class imbalance detected. Consider using class weights, oversampling, or data augmentation.")
        
        if imbalance_ratio > 5:
            recommendations.append("📊 Moderate to high imbalance. Consider stratified sampling for train/val/test splits.")
        
        if data_quality_score < 90:
            recommendations.append("🔍 Data quality issues detected. Review missing paths and duplicate samples.")
        
        total_samples = report['summary_statistics']['basic_stats']['total_samples']
        if total_samples < 1000:
            recommendations.append("📈 Small dataset size. Consider data augmentation and transfer learning.")
        
        if total_samples > 10000:
            recommendations.append("🚀 Large dataset detected. Consider using efficient training strategies and distributed training.")
        
        return recommendations


def main():
    """Main function for dataset inspection."""
    config = DatasetConfig()
    analyzer = DatasetAnalyzer(config)
    
    # First, inspect available data
    console.print("[bold blue]Inspecting available annotation data...[/bold blue]")
    data_info = analyzer.inspector.inspect_available_data()
    
    if not data_info['datasets']:
        console.print("[red]No datasets found! Please ensure the annotation API is running and has data.[/red]")
        return
    
    # Display available datasets
    table = Table(title="Available Datasets")
    table.add_column("Project", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Samples", style="yellow")
    table.add_column("Classes", style="blue")
    
    for dataset in data_info['datasets']:
        progress = dataset['progress']
        table.add_row(
            dataset['project_name'],
            dataset['dataset_name'],
            dataset['target_type'],
            str(progress.get('labeled', 0)),
            str(dataset['num_classes'])
        )
    
    console.print(table)
    
    # Analyze the first dataset
    if data_info['datasets']:
        dataset_id = data_info['datasets'][0]['dataset_id']
        console.print(f"\n[bold green]Analyzing dataset {dataset_id}...[/bold green]")
        
        report = analyzer.generate_comprehensive_report(dataset_id)
        console.print(f"\n[green]Analysis complete! Check the output directory for detailed visualizations.[/green]")


if __name__ == "__main__":
    main()