"""
optimization_analysis.py
Model Generalization Framework - Optimization Analysis Module
Author: Manvanth Sai

This module uses cross-validation to find optimal hyperparameters and 
demonstrates learning curves for understanding data requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from pathlib import Path
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set_context("notebook", font_scale=1.1)


class OptimizationAnalyzer:
    """Performs hyperparameter optimization using cross-validation."""
    
    def __init__(self, output_dir: Path, random_seed: int = 42):
        """
        Initialize the optimization analyzer.
        
        Args:
            output_dir: Directory to save visualizations
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def optimize_regularization_strength(self, dataset: Dict,
                                         polynomial_degree: int = 12,
                                         n_folds: int = 7) -> Dict:
        """
        Find optimal regularization strength using k-fold cross-validation.
        
        Args:
            dataset: Dictionary containing train/test data
            polynomial_degree: Degree of polynomial features
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing optimization results
        """
        # Combine train and test for cross-validation
        X_combined = np.vstack([dataset['X_train'], dataset['X_test']])
        y_combined = np.hstack([dataset['y_train'], dataset['y_test']])
        
        # Create pipeline
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=polynomial_degree)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        # Define parameter range (logarithmic scale)
        alpha_range = np.logspace(-6, 4, 70)
        
        print(f"Computing validation curves with {n_folds}-fold CV...")
        
        # Validation curve for Ridge
        train_scores_ridge, test_scores_ridge = validation_curve(
            pipeline, X_combined, y_combined,
            param_name="ridge__alpha",
            param_range=alpha_range,
            cv=n_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        
        # Calculate statistics
        train_mean_ridge = -np.mean(train_scores_ridge, axis=1)
        train_std_ridge = np.std(train_scores_ridge, axis=1)
        test_mean_ridge = -np.mean(test_scores_ridge, axis=1)
        test_std_ridge = np.std(test_scores_ridge, axis=1)
        
        # Find optimal alpha
        optimal_idx = np.argmin(test_mean_ridge)
        optimal_alpha = alpha_range[optimal_idx]
        best_cv_score = test_mean_ridge[optimal_idx]
        
        # Validation curve for Lasso (for comparison)
        pipeline.set_params(ridge=Lasso(max_iter=10000))
        print("Computing Lasso validation curves...")
        
        train_scores_lasso, test_scores_lasso = validation_curve(
            pipeline, X_combined, y_combined,
            param_name="ridge__alpha",
            param_range=alpha_range,
            cv=n_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        
        train_mean_lasso = -np.mean(train_scores_lasso, axis=1)
        test_mean_lasso = -np.mean(test_scores_lasso, axis=1)
        
        # Learning curves
        print("Computing learning curves...")
        pipeline.set_params(ridge=Ridge(alpha=optimal_alpha))
        
        train_sizes, train_scores_lc, test_scores_lc = learning_curve(
            pipeline, X_combined, y_combined,
            cv=n_folds,
            train_sizes=np.linspace(0.15, 1.0, 12),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=self.random_seed
        )
        
        train_mean_lc = -np.mean(train_scores_lc, axis=1)
        train_std_lc = np.std(train_scores_lc, axis=1)
        test_mean_lc = -np.mean(test_scores_lc, axis=1)
        test_std_lc = np.std(test_scores_lc, axis=1)
        
        # Generate visualizations
        self._create_optimization_visualizations(
            alpha_range, optimal_alpha, optimal_idx,
            train_mean_ridge, train_std_ridge, test_mean_ridge, test_std_ridge,
            train_mean_lasso, test_mean_lasso,
            train_sizes, train_mean_lc, train_std_lc, test_mean_lc, test_std_lc,
            n_folds, polynomial_degree
        )
        
        # Create interactive visualization
        self._create_interactive_plots(
            alpha_range, train_mean_ridge, test_mean_ridge,
            train_mean_lasso, test_mean_lasso, optimal_alpha
        )
        
        return {
            'optimal_alpha': optimal_alpha,
            'best_cv_score': best_cv_score,
            'alpha_range': alpha_range,
            'ridge_validation': (train_mean_ridge, test_mean_ridge),
            'lasso_validation': (train_mean_lasso, test_mean_lasso),
            'learning_curve': (train_sizes, train_mean_lc, test_mean_lc)
        }
    
    def _create_optimization_visualizations(self, alpha_range, optimal_alpha, optimal_idx,
                                           train_ridge, train_std_ridge, test_ridge, test_std_ridge,
                                           train_lasso, test_lasso,
                                           train_sizes, train_lc, train_std_lc, test_lc, test_std_lc,
                                           n_folds, degree) -> None:
        """Create comprehensive optimization visualizations."""
        
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
        
        # Plot 1: Ridge Validation Curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogx(alpha_range, train_ridge, label='Training Score',
                    color='#3498db', linewidth=2.5, alpha=0.85)
        ax1.fill_between(alpha_range, train_ridge - train_std_ridge,
                        train_ridge + train_std_ridge, alpha=0.2, color='#3498db')
        
        ax1.semilogx(alpha_range, test_ridge, label='CV Score',
                    color='#e74c3c', linewidth=2.5, alpha=0.85)
        ax1.fill_between(alpha_range, test_ridge - test_std_ridge,
                        test_ridge + test_std_ridge, alpha=0.2, color='#e74c3c')
        
        ax1.axvline(optimal_alpha, linestyle='--', color='#2ecc71',
                   linewidth=2.5, label=f'Optimal α: {optimal_alpha:.2e}')
        
        ax1.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax1.set_title(f'Ridge Validation Curve ({n_folds}-Fold CV)',
                     fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='best', framealpha=0.9, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Lasso Validation Curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.semilogx(alpha_range, train_lasso, label='Training Score',
                    color='#9b59b6', linewidth=2.5, alpha=0.85)
        ax2.semilogx(alpha_range, test_lasso, label='CV Score',
                    color='#f39c12', linewidth=2.5, alpha=0.85)
        
        ax2.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax2.set_title(f'Lasso Validation Curve ({n_folds}-Fold CV)',
                     fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='best', framealpha=0.9, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Ridge vs Lasso Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.semilogx(alpha_range, test_ridge, label='Ridge CV',
                    color='#e74c3c', linewidth=2.5, alpha=0.85)
        ax3.semilogx(alpha_range, test_lasso, label='Lasso CV',
                    color='#f39c12', linewidth=2.5, alpha=0.85, linestyle='--')
        ax3.axvline(optimal_alpha, linestyle=':', color='#2ecc71',
                   linewidth=2, alpha=0.7, label='Optimal α (Ridge)')
        
        ax3.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Cross-Validation MSE', fontsize=12, fontweight='bold')
        ax3.set_title('Ridge vs Lasso Performance',
                     fontsize=13, fontweight='bold', pad=10)
        ax3.legend(loc='best', framealpha=0.9, fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Learning Curve
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax4.plot(train_sizes, train_lc, 'o-', color='#3498db',
                linewidth=2.5, markersize=8, label='Training Score', alpha=0.85)
        ax4.fill_between(train_sizes, train_lc - train_std_lc,
                        train_lc + train_std_lc, alpha=0.2, color='#3498db')
        
        ax4.plot(train_sizes, test_lc, 's-', color='#e74c3c',
                linewidth=2.5, markersize=8, label='CV Score', alpha=0.85)
        ax4.fill_between(train_sizes, test_lc - test_std_lc,
                        test_lc + test_std_lc, alpha=0.2, color='#e74c3c')
        
        ax4.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax4.set_title(f'Learning Curve (α={optimal_alpha:.2e}, Degree={degree})',
                     fontsize=13, fontweight='bold', pad=10)
        ax4.legend(loc='best', framealpha=0.9, fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 5: Convergence Analysis
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Calculate rate of change (convergence)
        convergence_ridge = np.abs(np.diff(test_ridge))
        convergence_lasso = np.abs(np.diff(test_lasso))
        
        ax5.loglog(alpha_range[:-1], convergence_ridge, 'o-',
                  color='#e74c3c', linewidth=2, markersize=6,
                  label='Ridge Δ', alpha=0.85)
        ax5.loglog(alpha_range[:-1], convergence_lasso, 's-',
                  color='#f39c12', linewidth=2, markersize=6,
                  label='Lasso Δ', alpha=0.85)
        
        ax5.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('|ΔMSE| (Sensitivity)', fontsize=12, fontweight='bold')
        ax5.set_title('Parameter Sensitivity Analysis',
                     fontsize=13, fontweight='bold', pad=10)
        ax5.legend(loc='best', framealpha=0.9, fontsize=10)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('Hyperparameter Optimization via Cross-Validation\nAuthor: Manvanth Sai',
                    fontsize=16, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / 'hyperparameter_optimization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Optimization analysis visualization saved: {output_path}")
        print(f"\n✓ Optimal regularization strength: α = {optimal_alpha:.6f}")
        print(f"✓ Best cross-validation MSE: {test_ridge[optimal_idx]:.4f}")
    
    def _create_interactive_plots(self, alpha_range, train_ridge, test_ridge,
                                  train_lasso, test_lasso, optimal_alpha) -> None:
        """Create interactive Plotly visualization for exploration."""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Ridge Regularization', 'Lasso Regularization'),
            horizontal_spacing=0.15
        )
        
        # Ridge plot
        fig.add_trace(
            go.Scatter(x=alpha_range, y=train_ridge, mode='lines',
                      name='Ridge Train', line=dict(color='#3498db', width=3),
                      hovertemplate='α: %{x:.2e}<br>MSE: %{y:.4f}'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=alpha_range, y=test_ridge, mode='lines',
                      name='Ridge CV', line=dict(color='#e74c3c', width=3),
                      hovertemplate='α: %{x:.2e}<br>MSE: %{y:.4f}'),
            row=1, col=1
        )
        
        # Lasso plot
        fig.add_trace(
            go.Scatter(x=alpha_range, y=train_lasso, mode='lines',
                      name='Lasso Train', line=dict(color='#9b59b6', width=3),
                      hovertemplate='α: %{x:.2e}<br>MSE: %{y:.4f}'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=alpha_range, y=test_lasso, mode='lines',
                      name='Lasso CV', line=dict(color='#f39c12', width=3),
                      hovertemplate='α: %{x:.2e}<br>MSE: %{y:.4f}'),
            row=1, col=2
        )
        
        # Add optimal alpha line
        fig.add_vline(x=optimal_alpha, line_dash="dash", line_color="green",
                     annotation_text=f"Optimal α: {optimal_alpha:.2e}",
                     annotation_position="top right", row=1, col=1)
        
        # Update axes
        fig.update_xaxes(type="log", title_text="Regularization Strength (α)", row=1, col=1)
        fig.update_xaxes(type="log", title_text="Regularization Strength (α)", row=1, col=2)
        fig.update_yaxes(title_text="Mean Squared Error", row=1, col=1)
        fig.update_yaxes(title_text="Mean Squared Error", row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Validation Curves - Model Generalization Framework<br>"
                      "<sub>Author: Manvanth Sai</sub>",
            height=550,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12)
        )
        
        output_path = self.output_dir / 'interactive_validation_curves.html'
        fig.write_html(str(output_path))
        print(f"✓ Interactive visualization saved: {output_path}")
