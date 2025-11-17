"""
regularization_analysis.py
Model Generalization Framework - Regularization Analysis Module
Author: Manvanth Sai

This module compares different regularization techniques (Ridge, Lasso, ElasticNet)
and demonstrates how they control model complexity and improve generalization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path
from typing import List, Dict
import pandas as pd

sns.set_context("notebook", font_scale=1.1)


class RegularizationAnalyzer:
    """Compares and analyzes different regularization techniques."""
    
    def __init__(self, output_dir: Path, random_seed: int = 42):
        """
        Initialize the regularization analyzer.
        
        Args:
            output_dir: Directory to save visualizations
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def build_regularized_models(self, alpha_values: List[float]) -> Dict:
        """
        Construct a suite of regularized models with different strengths.
        
        Args:
            alpha_values: List of regularization strengths to test
            
        Returns:
            Dictionary of model instances
        """
        models = {
            'No Regularization': LinearRegression()
        }
        
        # Ridge (L2) models with different strengths
        for alpha in alpha_values:
            models[f'Ridge (α={alpha})'] = Ridge(alpha=alpha, random_state=self.random_seed)
        
        # Lasso (L1) models with different strengths
        for alpha in alpha_values:
            models[f'Lasso (α={alpha})'] = Lasso(alpha=alpha, random_state=self.random_seed, 
                                                  max_iter=10000)
        
        # ElasticNet (L1 + L2) with moderate strengths
        for alpha in alpha_values[:3]:
            models[f'ElasticNet (α={alpha})'] = ElasticNet(
                alpha=alpha, l1_ratio=0.5, random_state=self.random_seed, max_iter=10000
            )
        
        return models
    
    def compare_regularization_methods(self, dataset: Dict, 
                                       polynomial_degree: int = 15,
                                       alpha_range: List[float] = None) -> Dict:
        """
        Compare different regularization techniques on polynomial regression.
        
        Args:
            dataset: Dictionary containing train/test data
            polynomial_degree: Degree of polynomial features
            alpha_range: List of regularization strengths to test
            
        Returns:
            Dictionary containing comparison results
        """
        if alpha_range is None:
            alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        
        # Feature engineering
        poly_transform = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
        X_train_poly = poly_transform.fit_transform(X_train)
        X_test_poly = poly_transform.transform(X_test)
        
        # Standardization is crucial for regularization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_test_scaled = scaler.transform(X_test_poly)
        
        # Build model suite
        models = self.build_regularized_models(alpha_range)
        
        # Evaluate each model
        results = []
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Coefficient analysis
                if hasattr(model, 'coef_'):
                    n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
                    coef_l1_norm = np.sum(np.abs(model.coef_))
                    coef_l2_norm = np.sqrt(np.sum(model.coef_**2))
                else:
                    n_nonzero = coef_l1_norm = coef_l2_norm = np.nan
                
                results.append({
                    'Method': name,
                    'Train MSE': train_mse,
                    'Test MSE': test_mse,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Generalization Gap': test_mse - train_mse,
                    'Active Features': n_nonzero,
                    'L1 Norm': coef_l1_norm,
                    'L2 Norm': coef_l2_norm
                })
                
                trained_models[name] = model
                
            except Exception as e:
                print(f"Warning: {name} failed - {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Generate visualizations
        self._create_regularization_visualizations(
            dataset, poly_transform, scaler, trained_models, 
            results_df, polynomial_degree
        )
        
        # Find best model
        best_idx = results_df['Test MSE'].idxmin()
        best_method = results_df.loc[best_idx, 'Method']
        best_test_mse = results_df.loc[best_idx, 'Test MSE']
        
        return {
            'methods': list(trained_models.keys()),
            'results_table': results_df,
            'best_method': best_method,
            'best_test_mse': best_test_mse,
            'trained_models': trained_models
        }
    
    def _create_regularization_visualizations(self, dataset: Dict,
                                             poly_transform, scaler,
                                             models: Dict, results_df: pd.DataFrame,
                                             degree: int) -> None:
        """
        Create comprehensive regularization comparison visualizations.
        
        Args:
            dataset: Original dataset
            poly_transform: Polynomial feature transformer
            scaler: Feature scaler
            models: Dictionary of trained models
            results_df: DataFrame with performance metrics
            degree: Polynomial degree used
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        
        # Smooth prediction curve
        X_smooth = np.linspace(dataset['X_full'].min(), dataset['X_full'].max(), 400).reshape(-1, 1)
        X_smooth_poly = poly_transform.transform(X_smooth)
        X_smooth_scaled = scaler.transform(X_smooth_poly)
        y_true_smooth = 0.5 * X_smooth.ravel()**3 - 2.0 * X_smooth.ravel()**2 + X_smooth.ravel()
        
        # Plot 1: Ridge Regularization Effects
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.scatter(X_train, y_train, alpha=0.5, s=50, color='gray', label='Training Data')
        ax1.plot(X_smooth, y_true_smooth, 'k--', linewidth=2.5, label='True Function', alpha=0.6)
        
        ridge_models = {k: v for k, v in models.items() if 'Ridge' in k or 'No Reg' in k}
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ridge_models)))
        
        for (name, model), color in zip(ridge_models.items(), colors):
            y_pred = model.predict(X_smooth_scaled)
            ax1.plot(X_smooth, y_pred, linewidth=2.5, label=name, color=color, alpha=0.85)
        
        ax1.set_xlabel('x', fontsize=12, fontweight='bold')
        ax1.set_ylabel('y', fontsize=12, fontweight='bold')
        ax1.set_title(f'Ridge (L2) Regularization Effects (Degree {degree})',
                     fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Lasso Regularization Effects
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(X_train, y_train, alpha=0.5, s=50, color='gray', label='Training Data')
        ax2.plot(X_smooth, y_true_smooth, 'k--', linewidth=2.5, label='True Function', alpha=0.6)
        
        lasso_models = {k: v for k, v in models.items() if 'Lasso' in k}
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lasso_models)))
        
        for (name, model), color in zip(lasso_models.items(), colors):
            y_pred = model.predict(X_smooth_scaled)
            ax2.plot(X_smooth, y_pred, linewidth=2.5, label=name, color=color, alpha=0.85)
        
        ax2.set_xlabel('x', fontsize=12, fontweight='bold')
        ax2.set_ylabel('y', fontsize=12, fontweight='bold')
        ax2.set_title(f'Lasso (L1) Regularization Effects',
                     fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Performance Comparison
        ax3 = fig.add_subplot(gs[1, 0:2])
        display_models = results_df.head(10)  # Show top 10 for clarity
        
        x_pos = np.arange(len(display_models))
        width = 0.35
        
        ax3.bar(x_pos - width/2, display_models['Train MSE'], width,
               label='Train MSE', color='#3498db', alpha=0.8)
        ax3.bar(x_pos + width/2, display_models['Test MSE'], width,
               label='Test MSE', color='#e74c3c', alpha=0.8)
        
        ax3.set_xlabel('Regularization Method', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Mean Squared Error (log scale)', fontsize=11, fontweight='bold')
        ax3.set_title('Training vs Test Performance', fontsize=13, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(display_models['Method'], rotation=45, ha='right', fontsize=8)
        ax3.legend(framealpha=0.9, fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_yscale('log')
        
        # Plot 4: Feature Selection (Sparsity)
        ax4 = fig.add_subplot(gs[1, 2])
        sparsity_data = results_df.dropna(subset=['Active Features']).head(10)
        
        bars = ax4.barh(range(len(sparsity_data)), sparsity_data['Active Features'],
                       color=plt.cm.RdYlGn_r(sparsity_data['Active Features'] / 
                                            sparsity_data['Active Features'].max()))
        
        ax4.set_yticks(range(len(sparsity_data)))
        ax4.set_yticklabels(sparsity_data['Method'], fontsize=9)
        ax4.set_xlabel('Number of Active Features', fontsize=11, fontweight='bold')
        ax4.set_title('Feature Selection Capability', fontsize=13, fontweight='bold', pad=10)
        ax4.axvline(x=poly_transform.n_output_features_, color='red', 
                   linestyle='--', linewidth=2, alpha=0.7, label='Total Features')
        ax4.legend(framealpha=0.9, fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Plot 5: Coefficient Magnitude Comparison
        ax5 = fig.add_subplot(gs[2, 0:2])
        
        # Select representative models
        coef_models = ['No Regularization'] + \
                     [k for k in models.keys() if 'Ridge' in k][:3] + \
                     [k for k in models.keys() if 'Lasso' in k][:3]
        
        for name in coef_models:
            if name in models and hasattr(models[name], 'coef_'):
                coefs = np.abs(models[name].coef_)
                ax5.plot(coefs, alpha=0.7, linewidth=2, label=name, marker='o', markersize=3)
        
        ax5.set_xlabel('Coefficient Index', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Absolute Coefficient Value (log scale)', fontsize=11, fontweight='bold')
        ax5.set_title('Coefficient Shrinkage Patterns', fontsize=13, fontweight='bold', pad=10)
        ax5.set_yscale('log')
        ax5.legend(loc='best', fontsize=8, framealpha=0.9)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 6: Generalization Gap
        ax6 = fig.add_subplot(gs[2, 2])
        gap_data = results_df.dropna(subset=['Generalization Gap']).head(10)
        colors = ['#2ecc71' if gap < 1 else '#f39c12' if gap < 3 else '#e74c3c' 
                 for gap in gap_data['Generalization Gap']]
        
        bars = ax6.bar(range(len(gap_data)), gap_data['Generalization Gap'],
                      color=colors, alpha=0.75, edgecolor='white', linewidth=2)
        
        ax6.set_xticks(range(len(gap_data)))
        ax6.set_xticklabels(gap_data['Method'], rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Test MSE - Train MSE', fontsize=11, fontweight='bold')
        ax6.set_title('Generalization Gap (Lower is Better)', 
                     fontsize=13, fontweight='bold', pad=10)
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.suptitle('Regularization Techniques Comparison\nAuthor: Manvanth Sai',
                    fontsize=16, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / 'regularization_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Regularization comparison visualization saved: {output_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("REGULARIZATION PERFORMANCE SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
