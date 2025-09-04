#!/usr/bin/env python3
"""
Distributed Training Script
Launch distributed training across multiple symbols and devices
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from services.distributed_compute import (
        create_distributed_training_pipeline,
        DistributedTrainingManager,
        ClusterManager
    )
    DISTRIBUTED_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Distributed training imports not available: {e}")
    DISTRIBUTED_IMPORTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed Stock AI Training System')
    
    # Symbols to train
    parser.add_argument('--symbols', type=str, nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                       help='Stock symbols to train models for')
    
    parser.add_argument('--symbols-file', type=str,
                       help='File containing stock symbols (one per line)')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble', 
                               'nextgen_ensemble', 'advanced_ensemble'],
                       help='Model type to train')
    
    parser.add_argument('--backend', type=str, default='tensorflow',
                       choices=['tensorflow', 'pytorch'],
                       help='Training backend to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--feature-level', type=str, default='intelligent',
                       choices=['standard', 'advanced', 'intelligent', 'all'],
                       help='Feature engineering level')
    
    # Distributed options
    parser.add_argument('--force-sequential', action='store_true',
                       help='Force sequential training (disable distributed)')
    
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of workers (auto-detect if not set)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/distributed',
                       help='Directory to save results')
    
    parser.add_argument('--save-models', action='store_true', default=True,
                       help='Save trained models')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def load_symbols_from_file(filepath: str) -> list:
    """Load symbols from file"""
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except Exception as e:
        logger.error(f"Failed to load symbols from {filepath}: {e}")
        return []

def setup_output_directory(output_dir: str) -> str:
    """Setup output directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_output_dir = f"{output_dir}/run_{timestamp}"
    
    os.makedirs(full_output_dir, exist_ok=True)
    logger.info(f"Output directory: {full_output_dir}")
    
    return full_output_dir

def save_results(results: dict, output_dir: str):
    """Save training results"""
    import json
    
    results_file = os.path.join(output_dir, 'training_results.json')
    
    try:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Save summary
        summary_file = os.path.join(output_dir, 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Distributed Training Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Symbols: {results['training']['total_symbols']}\n")
            f.write(f"Successful: {results['training']['successful']}\n")
            f.write(f"Failed: {results['training']['failed']}\n")
            f.write(f"Training Method: {results['training']['training_method']}\n")
            f.write(f"Backend: {results['pipeline']['backend']}\n")
            f.write(f"Duration: {results['pipeline']['duration_seconds']:.2f}s\n")
            
            if 'summary' in results['training']:
                summary = results['training']['summary']
                f.write(f"\nPerformance Metrics:\n")
                f.write(f"Average MSE: {summary.get('avg_mse', 0):.6f}\n")
                f.write(f"Average MAE: {summary.get('avg_mae', 0):.6f}\n")
                f.write(f"Average RMSE: {summary.get('avg_mse', 0)**0.5:.6f}\n")
                f.write(f"Total Training Time: {summary.get('total_training_time', 0):.2f}s\n")
                f.write(f"Worker Count: {summary.get('worker_count', 1)}\n")
            
            if 'performance' in results:
                perf = results['performance']
                f.write(f"\nEfficiency Metrics:\n")
                f.write(f"Symbols/Second: {perf.get('symbols_per_second', 0):.2f}\n")
                f.write(f"Avg Time/Symbol: {perf.get('avg_training_time_per_symbol', 0):.2f}s\n")
                f.write(f"Speedup Factor: {perf.get('speedup_factor', 1.0):.2f}x\n")
        
        logger.info(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def print_system_info():
    """Print system information"""
    if not DISTRIBUTED_IMPORTS_AVAILABLE:
        print("❌ Distributed training not available")
        return
    
    try:
        # Get cluster information
        cluster_manager = ClusterManager()
        cluster_status = cluster_manager.get_cluster_status()
        
        print("\n🖥️  System Information")
        print("=" * 50)
        
        # Local resources
        local_info = cluster_status['clusters']['local']
        print(f"CPU Cores: {local_info['cpu_count']}")
        print(f"Memory: {local_info['memory_gb']:.1f} GB")
        
        # GPU information
        training_manager = DistributedTrainingManager()
        devices = training_manager.devices
        print(f"GPUs: {devices['gpu_count']}")
        
        if devices['gpu_memory']:
            for gpu in devices['gpu_memory']:
                print(f"  - {gpu}")
        
        # Ray cluster information
        if 'ray_detailed' in cluster_status:
            ray_info = cluster_status['ray_detailed']
            if 'error' not in ray_info:
                print(f"\n🚀 Ray Cluster:")
                print(f"Alive Nodes: {ray_info['alive_nodes']}")
                print(f"Total CPUs: {ray_info['cluster_resources'].get('CPU', 0)}")
                print(f"Total GPUs: {ray_info['cluster_resources'].get('GPU', 0)}")
        
        print(f"\nAvailable Backends: {devices['available_backends']}")
        print()
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")

def main():
    """Main distributed training function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Stock AI Distributed Training System")
    print("=" * 50)
    
    if not DISTRIBUTED_IMPORTS_AVAILABLE:
        print("❌ Distributed training components not available")
        print("Please install required dependencies:")
        print("  pip install ray joblib tensorflow torch")
        return 1
    
    # Print system information
    print_system_info()
    
    # Load symbols
    symbols = args.symbols
    if args.symbols_file:
        file_symbols = load_symbols_from_file(args.symbols_file)
        if file_symbols:
            symbols = file_symbols
    
    if not symbols:
        logger.error("No symbols provided for training")
        return 1
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Training configuration
    logger.info(f"Training Configuration:")
    logger.info(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Feature Level: {args.feature_level}")
    logger.info(f"  Force Sequential: {args.force_sequential}")
    logger.info(f"  Output: {output_dir}")
    
    try:
        # Override distributed settings if forced sequential
        if args.force_sequential:
            os.environ['FORCE_SEQUENTIAL'] = '1'
        
        # Run distributed training pipeline
        logger.info("🎯 Starting distributed training pipeline...")
        
        results = create_distributed_training_pipeline(
            symbols=symbols,
            model_type=args.model,
            backend=args.backend,
            epochs=args.epochs,
            batch_size=args.batch_size,
            feature_level=args.feature_level
        )
        
        # Save results
        save_results(results, output_dir)
        
        # Print summary
        print("\n✅ Training Completed!")
        print("=" * 30)
        print(f"Total Symbols: {results['training']['total_symbols']}")
        print(f"Successful: {results['training']['successful']}")
        print(f"Failed: {results['training']['failed']}")
        print(f"Method: {results['training']['training_method']}")
        print(f"Duration: {results['pipeline']['duration_seconds']:.2f}s")
        
        if 'summary' in results['training']:
            summary = results['training']['summary']
            avg_rmse = summary.get('avg_mse', 0) ** 0.5
            print(f"Average RMSE: {avg_rmse:.6f}")
            print(f"Worker Count: {summary.get('worker_count', 1)}")
        
        if 'performance' in results:
            perf = results['performance']
            print(f"Speedup: {perf.get('speedup_factor', 1.0):.2f}x")
        
        print(f"\nResults saved to: {output_dir}")
        
        # Success rate check
        success_rate = results['training']['successful'] / results['training']['total_symbols']
        if success_rate < 0.5:
            logger.warning(f"Low success rate: {success_rate:.1%}")
            return 2  # Partial failure
        
        return 0  # Success
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())