"""
complexity_analysis.py
Model Generalization Framework - Complexity Analysis Module
Author: Manvanth Sai

This module analyzes how model complexity affects generalization performance.
It demonstrates the bias-variance tradeoff through polynomial regression.
"""

""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
from typing import List, Dict

sns.set_context("notebook", font_scale=1.15)


class ComplexityAnalyzer:
    """Analyzes the relationship between model complexity and generalization."""
    
    def __init__(self, output_dir: Path, random_seed: int = 42):
        """
        Initialize the complexity analyzer.
        
        Args:
            output_dir: Directory to save visualizations
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def fit_polynomial(self, X_train: np.ndarray, y_train: np.ndarray, 
                      degree: int) -> tuple:
        """
        Fit a polynomial regression model of specified degree.
        
        Args:
            X_train: Training features
            y_train: Training targets
            degree: Polynomial degree
            
        Returns:
            Tuple of (fitted_model, polynomial_transformer)
        """
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        X_train_poly = poly_features.fit_transform(X_train)
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train_poly, y_train)
        
        return model, poly_features
    
    def compute_performance_metrics(self, model, poly_features, 
                                    X_train, y_train, X_test, y_test, 
                                    degree: int) -> Dict:
        """
        Calculate comprehensive performance metrics for a fitted model.
        
        Args:
            model: Fitted regression model
            poly_features: Polynomial feature transformer
            X_train, y_train: Training data
            X_test, y_test: Test data
            degree: Polynomial degree
            
        Returns:
            Dictionary of performance metrics
        """
        # Transform features
        X_train_poly = poly_features.transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # Predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Overfitting indicator: ratio of test to train error
        generalization_gap = test_mse / train_mse if train_mse > 1e-10 else np.inf
        
        return {
            'degree': degree,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'n_params': len(model.coef_),
            'generalization_gap': generalization_gap
        }
    
    def analyze_polynomial_fits(self, dataset: Dict, degrees: List[int]) -> Dict:
        """
        Analyze polynomial models across different degrees to demonstrate
        the bias-variance tradeoff.
        
        Args:
            dataset: Dictionary containing train/test data
            degrees: List of polynomial degrees to evaluate
            
        Returns:
            Dictionary containing analysis results
        """
        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        
        results = []
        fitted_models = {}
        
        # Fit and evaluate each polynomial degree
        for degree in degrees:
            model, poly_features = self.fit_polynomial(X_train, y_train, degree)
            metrics = self.compute_performance_metrics(
                model, poly_features, X_train, y_train, X_test, y_test, degree
            )
            results.append(metrics)
            fitted_models[degree] = (model, poly_features)
        
        # Generate visualizations
        self._create_complexity_visualizations(dataset, fitted_models, results)
        
        # Compile results
        output = {
            'degrees': [r['degree'] for r in results],
            'train_errors': [r['train_mse'] for r in results],
            'test_errors': [r['test_mse'] for r in results],
            'train_r2': [r['train_r2'] for r in results],
            'test_r2': [r['test_r2'] for r in results],
            'generalization_gaps': [r['generalization_gap'] for r in results],
            'detailed_metrics': results
        }
        
        return output
    
    def _create_complexity_visualizations(self, dataset: Dict, 
                                         models: Dict, 
                                         metrics: List[Dict]) -> None:
        """
        Create comprehensive visualizations for complexity analysis.
        
        Args:
            dataset: Original dataset
            models: Dictionary of fitted models
            metrics: List of performance metrics
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Prepare plot data
        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        X_full = dataset['X_full']
        y_true = dataset['y_true']
        
        X_smooth = np.linspace(X_full.min(), X_full.max(), 400).reshape(-1, 1)
        y_true_smooth = 0.5 * X_smooth.ravel()**3 - 2.0 * X_smooth.ravel()**2 + X_smooth.ravel()
        
        # Select representative degrees for visualization
        degrees_to_plot = sorted(models.keys())
        if len(degrees_to_plot) > 3:
            # Show low, medium, high complexity
            selected = [degrees_to_plot[0], 
                       degrees_to_plot[len(degrees_to_plot)//2],
                       degrees_to_plot[-1]]
        else:
            selected = degrees_to_plot
        
        # Plots 1-3: Individual polynomial fits
        for idx, degree in enumerate(selected):
            ax = fig.add_subplot(gs[0, idx])
            model, poly_features = models[degree]
            
            # Generate smooth predictions
            X_smooth_poly = poly_features.transform(X_smooth)
            y_pred_smooth = model.predict(X_smooth_poly)
            
            # Plot data and predictions
            ax.scatter(X_train, y_train, alpha=0.6, s=60, color='#3498db', 
                      label='Training', edgecolors='white', linewidth=1.5)
            ax.scatter(X_test, y_test, alpha=0.6, s=60, color='#e74c3c',
                      label='Test', edgecolors='white', linewidth=1.5)
            ax.plot(X_smooth, y_true_smooth, 'k--', linewidth=2.5, 
                   label='True Function', alpha=0.6)
            ax.plot(X_smooth, y_pred_smooth, color='#2ecc71', linewidth=3,
                   label='Model Prediction', alpha=0.85)
            
            # Get metrics for this degree
            metric = next(m for m in metrics if m['degree'] == degree)
            
            ax.set_title(f'Polynomial Degree {degree}\n'
                        f'Train MSE: {metric["train_mse"]:.3f} | '
                        f'Test MSE: {metric["test_mse"]:.3f}',
                        fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('x', fontsize=11, fontweight='bold')
            ax.set_ylabel('y', fontsize=11, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([y_train.min() - 5, y_train.max() + 5])
        
        # Plot 4: Bias-Variance Tradeoff
        ax4 = fig.add_subplot(gs[1, 0])
        degrees = [m['degree'] for m in metrics]
        train_mse = [m['train_mse'] for m in metrics]
        test_mse = [m['test_mse'] for m in metrics]
        
        ax4.plot(degrees, train_mse, 'o-', linewidth=3, markersize=9,
                color='#3498db', label='Training Error', alpha=0.8)
        ax4.plot(degrees, test_mse, 's-', linewidth=3, markersize=9,
                color='#e74c3c', label='Test Error', alpha=0.8)
        
        # Mark the optimal point
        optimal_idx = np.argmin(test_mse)
        ax4.plot(degrees[optimal_idx], test_mse[optimal_idx], 'g*', 
                markersize=20, label='Optimal Complexity', zorder=5)
        
        ax4.set_xlabel('Model Complexity (Polynomial Degree)', 
                      fontsize=11, fontweight='bold')
        ax4.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
        ax4.set_title('Bias-Variance Tradeoff', fontsize=13, fontweight='bold', pad=10)
        ax4.legend(framealpha=0.9, fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_yscale('log')
        
        # Plot 5: R² Score Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        train_r2 = [m['train_r2'] for m in metrics]
        test_r2 = [m['test_r2'] for m in metrics]
        
        ax5.plot(degrees, train_r2, 'o-', linewidth=3, markersize=9,
                color='#9b59b6', label='Training R²', alpha=0.8)
        ax5.plot(degrees, test_r2, 's-', linewidth=3, markersize=9,
                color='#f39c12', label='Test R²', alpha=0.8)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax5.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
        ax5.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax5.set_title('Goodness of Fit Analysis', fontsize=13, fontweight='bold', pad=10)
        ax5.legend(framealpha=0.9, fontsize=10)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 6: Overfitting Severity
        ax6 = fig.add_subplot(gs[1, 2])
        gaps = [m['generalization_gap'] for m in metrics]
        colors = ['#2ecc71' if g < 2 else '#f39c12' if g < 5 else '#e74c3c' for g in gaps]
        
        bars = ax6.bar(degrees, gaps, color=colors, alpha=0.75, 
                      edgecolor='white', linewidth=2)
        ax6.axhline(y=1, color='black', linestyle='--', linewidth=2.5, 
                   alpha=0.7, label='Perfect Generalization')
        
        ax6.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Test Error / Train Error', fontsize=11, fontweight='bold')
        ax6.set_title('Overfitting Severity Index', fontsize=13, fontweight='bold', pad=10)
        ax6.legend(framealpha=0.9, fontsize=10)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Plot 7: Model Complexity vs Performance
        ax7 = fig.add_subplot(gs[2, 0])
        n_params = [m['n_params'] for m in metrics]
        
        scatter = ax7.scatter(n_params, test_mse, c=degrees, s=180, 
                             cmap='viridis', alpha=0.7, edgecolors='white', 
                             linewidth=2)
        ax7.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Test MSE (log scale)', fontsize=11, fontweight='bold')
        ax7.set_title('Model Capacity vs Performance', fontsize=13, fontweight='bold', pad=10)
        ax7.set_yscale('log')
        ax7.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Polynomial Degree', fontsize=10)
        
        # Plot 8: Error Comparison (MAE)
        ax8 = fig.add_subplot(gs[2, 1])
        train_mae = [m['train_mae'] for m in metrics]
        test_mae = [m['test_mae'] for m in metrics]
        
        x_pos = np.arange(len(degrees))
        width = 0.35
        
        ax8.bar(x_pos - width/2, train_mae, width, label='Train MAE',
               color='#3498db', alpha=0.75, edgecolor='white', linewidth=1)
        ax8.bar(x_pos + width/2, test_mae, width, label='Test MAE',
               color='#e74c3c', alpha=0.75, edgecolor='white', linewidth=1)
        
        ax8.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
        ax8.set_title('Prediction Error Magnitude', fontsize=13, fontweight='bold', pad=10)
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(degrees)
        ax8.legend(framealpha=0.9, fontsize=10)
        ax8.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Plot 9: Performance Summary Table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        table_data = []
        for m in metrics:
            table_data.append([
                f"{m['degree']}",
                f"{m['test_mse']:.3f}",
                f"{m['test_r2']:.3f}",
                f"{m['n_params']}"
            ])
        
        table = ax9.table(cellText=table_data,
                         colLabels=['Degree', 'Test MSE', 'Test R²', 'Params'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style the header
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best model
        best_idx = np.argmin(test_mse)
        for j in range(4):
            table[(best_idx + 1, j)].set_facecolor('#2ecc71')
            table[(best_idx + 1, j)].set_text_props(weight='bold')
        
        ax9.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
        
        # Overall title
        plt.suptitle('Model Complexity Analysis & Generalization Performance\nAuthor: Manvanth Sai',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = self.output_dir / 'complexity_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Complexity analysis visualization saved: {output_path}")
