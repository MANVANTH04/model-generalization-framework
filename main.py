"""
Advanced Machine Learning Regularization Framework
A comprehensive toolkit for understanding model complexity and generalization
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class MLFrameworkOrchestrator:
    """Orchestrates the execution of ML demonstrations with enhanced logging and error handling."""
    
    def __init__(self, output_dir: str = "results", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self, level: str) -> None:
        """Configure structured logging system."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.output_dir / 'framework.log')
            ]
        )
    
    def execute_pipeline(self) -> bool:
        """Execute complete ML demonstration pipeline with error recovery."""
        self.logger.info("="*70)
        self.logger.info("🚀 Advanced ML Regularization Framework Initialized")
        self.logger.info("="*70)
        
        pipeline_stages = [
            ("Model Complexity Analysis", self._stage_complexity),
            ("Regularization Techniques Benchmark", self._stage_regularization),
            ("Hyperparameter Optimization Suite", self._stage_optimization)
        ]
        
        results = {}
        for stage_name, stage_func in pipeline_stages:
            try:
                self.logger.info(f"\n📊 Executing: {stage_name}")
                self.logger.info("-" * 70)
                result = stage_func()
                results[stage_name] = result
                self.logger.info(f"✅ Completed: {stage_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed: {stage_name} - {str(e)}", exc_info=True)
                return False
        
        self._generate_summary_report(results)
        return True
    
    def _stage_complexity(self) -> dict:
        """Execute model complexity and overfitting analysis."""
        from modules.complexity_analyzer import ComplexityAnalyzer
        
        analyzer = ComplexityAnalyzer(output_dir=self.output_dir)
        data_bundle = analyzer.generate_synthetic_dataset(
            n_samples=150,
            noise_level=1.2,
            test_ratio=0.25
        )
        
        metrics = analyzer.analyze_model_complexity(
            data_bundle,
            degree_range=[1, 3, 5, 10, 20],
            generate_plots=True
        )
        
        self.logger.info(f"Analyzed {len(metrics)} polynomial degrees")
        return {"metrics": metrics, "data": data_bundle}
    
    def _stage_regularization(self) -> dict:
        """Execute regularization techniques comparison."""
        from modules.regularization_engine import RegularizationEngine
        from modules.complexity_analyzer import ComplexityAnalyzer
        
        # Get data from previous stage
        analyzer = ComplexityAnalyzer(output_dir=self.output_dir)
        data_bundle = analyzer.generate_synthetic_dataset(
            n_samples=150,
            noise_level=1.2,
            test_ratio=0.25
        )
        
        engine = RegularizationEngine(output_dir=self.output_dir)
        comparison = engine.comprehensive_comparison(
            data_bundle,
            polynomial_degree=18,
            alpha_values=[0.001, 0.01, 0.1, 1.0, 10.0]
        )
        
        self.logger.info(f"Compared {len(comparison['models'])} regularization techniques")
        return comparison
    
    def _stage_optimization(self) -> dict:
        """Execute hyperparameter optimization with cross-validation."""
        from modules.optimization_suite import OptimizationSuite
        from modules.complexity_analyzer import ComplexityAnalyzer
        
        analyzer = ComplexityAnalyzer(output_dir=self.output_dir)
        data_bundle = analyzer.generate_synthetic_dataset(
            n_samples=200,
            noise_level=1.0,
            test_ratio=0.2
        )
        
        optimizer = OptimizationSuite(output_dir=self.output_dir)
        results = optimizer.optimize_with_cross_validation(
            data_bundle,
            polynomial_degree=15,
            cv_folds=7,
            n_alpha_points=60
        )
        
        self.logger.info(f"Optimal α: {results['best_alpha']:.6f}")
        self.logger.info(f"Best CV Score: {results['best_score']:.4f}")
        return results
    
    def _generate_summary_report(self, results: dict) -> None:
        """Generate comprehensive execution summary."""
        self.logger.info("\n" + "="*70)
        self.logger.info("📈 EXECUTION SUMMARY")
        self.logger.info("="*70)
        
        for stage, result in results.items():
            self.logger.info(f"\n✓ {stage}")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  • {key}: {value}")
        
        self.logger.info(f"\n📁 Output Directory: {self.output_dir.absolute()}")
        output_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.html"))
        self.logger.info(f"📊 Generated {len(output_files)} visualization files")
        
        for file in sorted(output_files):
            self.logger.info(f"  • {file.name}")
        
        self.logger.info("\n" + "="*70)
        self.logger.info("✨ Framework execution completed successfully!")
        self.logger.info("="*70)


def main():
    """Main entry point for the ML Regularization Framework."""
    try:
        orchestrator = MLFrameworkOrchestrator(
            output_dir="results",
            log_level="INFO"
        )
        success = orchestrator.execute_pipeline()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
