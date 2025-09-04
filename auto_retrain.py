#!/usr/bin/env python3
"""
Stock AI Automation Service
Manages automated retraining, monitoring, and maintenance tasks
"""

import os
import sys
import json
import argparse
import signal
import logging
from pathlib import Path
from typing import List

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from services.auto_training import (
        AutomatedRetrainingPipeline,
        RetrainingConfig,
        create_default_config
    )
    AUTOMATION_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Automation imports not available: {e}")
    AUTOMATION_IMPORTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global pipeline
    if pipeline:
        print(f"\n🛑 Received signal {signum}, shutting down...")
        pipeline.stop()
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stock AI Automation Service')
    
    # Action commands
    parser.add_argument('action', choices=['start', 'monitor', 'retrain', 'status', 'config'],
                       help='Action to perform')
    
    # Symbols configuration
    parser.add_argument('--symbols', type=str, nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                       help='Stock symbols to monitor and retrain')
    
    parser.add_argument('--symbols-file', type=str,
                       help='File containing stock symbols (one per line)')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble', 
                               'nextgen_ensemble', 'advanced_ensemble'],
                       help='Model type to use')
    
    parser.add_argument('--feature-level', type=str, default='intelligent',
                       choices=['standard', 'advanced', 'intelligent', 'all'],
                       help='Feature engineering level')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Monitoring thresholds
    parser.add_argument('--performance-threshold', type=float, default=0.15,
                       help='RMSE threshold for triggering retraining')
    
    parser.add_argument('--data-freshness-days', type=int, default=30,
                       help='Days after which data is considered stale')
    
    # Schedule
    parser.add_argument('--schedule', type=str, default='0 2 * * 0',
                       help='Cron schedule for automated retraining')
    
    # Distributed training
    parser.add_argument('--disable-distributed', action='store_true',
                       help='Disable distributed training')
    
    parser.add_argument('--max-concurrent', type=int, default=4,
                       help='Maximum concurrent retraining jobs')
    
    # Notifications
    parser.add_argument('--enable-email', action='store_true',
                       help='Enable email notifications')
    
    parser.add_argument('--email-smtp-server', type=str, default='smtp.gmail.com',
                       help='SMTP server for email notifications')
    
    parser.add_argument('--email-smtp-port', type=int, default=587,
                       help='SMTP port for email notifications')
    
    parser.add_argument('--email-username', type=str,
                       help='SMTP username for email notifications')
    
    parser.add_argument('--email-password', type=str,
                       help='SMTP password for email notifications')
    
    parser.add_argument('--email-recipients', type=str, nargs='+',
                       help='Email recipients for notifications')
    
    # Configuration file
    parser.add_argument('--config-file', type=str,
                       help='Load configuration from JSON file')
    
    parser.add_argument('--save-config', type=str,
                       help='Save configuration to JSON file')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon process')
    
    return parser.parse_args()

def load_symbols_from_file(filepath: str) -> List[str]:
    """Load symbols from file"""
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except Exception as e:
        logger.error(f"Failed to load symbols from {filepath}: {e}")
        return []

def create_config_from_args(args) -> RetrainingConfig:
    """Create RetrainingConfig from command line arguments"""
    
    # Load symbols
    symbols = args.symbols
    if args.symbols_file:
        file_symbols = load_symbols_from_file(args.symbols_file)
        if file_symbols:
            symbols = file_symbols
    
    # Create configuration
    config = RetrainingConfig(
        symbols=symbols,
        model_type=args.model,
        feature_level=args.feature_level,
        epochs=args.epochs,
        batch_size=args.batch_size,
        schedule_cron=args.schedule,
        performance_threshold=args.performance_threshold,
        data_freshness_days=args.data_freshness_days,
        performance_window_days=7,
        min_samples_for_evaluation=10,
        use_distributed_training=not args.disable_distributed,
        max_concurrent_retrains=args.max_concurrent,
        email_notifications=args.enable_email,
        email_smtp_server=args.email_smtp_server,
        email_smtp_port=args.email_smtp_port,
        email_username=args.email_username or "",
        email_password=args.email_password or "",
        email_recipients=args.email_recipients or [],
        model_backup_days=30,
        log_retention_days=90
    )
    
    return config

def load_config_from_file(filepath: str) -> RetrainingConfig:
    """Load configuration from JSON file"""
    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return RetrainingConfig(**config_dict)
        
    except Exception as e:
        logger.error(f"Failed to load config from {filepath}: {e}")
        raise

def save_config_to_file(config: RetrainingConfig, filepath: str):
    """Save configuration to JSON file"""
    try:
        config_dict = config.__dict__.copy()
        
        # Make serializable
        if 'email_recipients' in config_dict and config_dict['email_recipients'] is None:
            config_dict['email_recipients'] = []
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {filepath}: {e}")

def action_start(config: RetrainingConfig, daemon: bool = False):
    """Start the automation service"""
    global pipeline
    
    if not AUTOMATION_IMPORTS_AVAILABLE:
        print("❌ Automation components not available")
        return 1
    
    print("🚀 Starting Stock AI Automation Service")
    print("=" * 50)
    print(f"Monitoring: {len(config.symbols)} symbols")
    print(f"Model Type: {config.model_type}")
    print(f"Feature Level: {config.feature_level}")
    print(f"Schedule: {config.schedule_cron}")
    print(f"Performance Threshold: {config.performance_threshold}")
    print(f"Distributed Training: {config.use_distributed_training}")
    
    if config.email_notifications:
        print(f"Email Notifications: Enabled ({len(config.email_recipients)} recipients)")
    else:
        print("Email Notifications: Disabled")
    
    print("=" * 50)
    
    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create and start pipeline
        pipeline = AutomatedRetrainingPipeline(config)
        pipeline.start()
        
        print("✅ Automation service started successfully!")
        print("Press Ctrl+C to stop the service")
        
        if daemon:
            # Run as daemon
            import time
            try:
                while pipeline.is_running:
                    time.sleep(60)
            except KeyboardInterrupt:
                pass
        else:
            # Interactive mode
            while True:
                try:
                    command = input("\nEnter command (status/retrain/stop/help): ").strip().lower()
                    
                    if command == 'status':
                        status = pipeline.get_status()
                        print(f"Running: {status['is_running']}")
                        print(f"Monitored Symbols: {status['monitored_symbols']}")
                        print(f"Need Retraining: {status['symbols_needing_retraining']}")
                    
                    elif command == 'retrain':
                        print("Triggering manual retraining...")
                        result = pipeline.force_retrain_all()
                        successful = result.get('successful', 0)
                        total = len(config.symbols)
                        print(f"Retraining completed: {successful}/{total} models")
                    
                    elif command == 'stop':
                        break
                    
                    elif command == 'help':
                        print("Available commands:")
                        print("  status  - Show current status")
                        print("  retrain - Trigger manual retraining")
                        print("  stop    - Stop the service")
                        print("  help    - Show this help")
                    
                    else:
                        print("Unknown command. Type 'help' for available commands.")
                
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
        
        return 0
        
    except Exception as e:
        logger.error(f"Automation service failed: {e}")
        return 1
    
    finally:
        if pipeline:
            pipeline.stop()

def action_monitor(config: RetrainingConfig):
    """Run performance monitoring"""
    if not AUTOMATION_IMPORTS_AVAILABLE:
        print("❌ Automation components not available")
        return 1
    
    print("🔍 Running Performance Monitoring")
    print("=" * 40)
    
    try:
        pipeline = AutomatedRetrainingPipeline(config)
        performances = pipeline.monitor_performance()
        
        print(f"\nPerformance Summary:")
        print(f"Total Models: {len(performances)}")
        
        needs_retraining = [p for p in performances if p.needs_retraining]
        print(f"Need Retraining: {len(needs_retraining)}")
        
        if needs_retraining:
            print("\nModels Requiring Retraining:")
            for perf in needs_retraining:
                print(f"  {perf.symbol}: RMSE={perf.rmse:.4f}, Reason={perf.retraining_reason}")
        
        good_performance = [p for p in performances if not p.needs_retraining]
        if good_performance:
            print("\nWell-Performing Models:")
            for perf in good_performance:
                print(f"  {perf.symbol}: RMSE={perf.rmse:.4f}, Accuracy={perf.directional_accuracy:.1%}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        return 1

def action_retrain(config: RetrainingConfig):
    """Run manual retraining"""
    if not AUTOMATION_IMPORTS_AVAILABLE:
        print("❌ Automation components not available")
        return 1
    
    print("🔄 Running Manual Retraining")
    print("=" * 35)
    
    try:
        pipeline = AutomatedRetrainingPipeline(config)
        
        # First check which models need retraining
        performances = pipeline.monitor_performance()
        needs_retraining = [p.symbol for p in performances if p.needs_retraining]
        
        if not needs_retraining:
            print("✅ No models require retraining based on current performance")
            return 0
        
        print(f"Retraining {len(needs_retraining)} models: {needs_retraining}")
        
        # Run retraining
        result = pipeline.executor.retrain_batch(needs_retraining)
        
        successful = result.get('successful', 0)
        total = result.get('total', 0)
        
        print(f"\n✅ Retraining completed: {successful}/{total} models successfully retrained")
        
        if 'results' in result:
            for r in result['results']:
                status = "✅" if r.get('status') == 'success' else "❌"
                symbol = r.get('symbol', 'Unknown')
                rmse = r.get('rmse', 0)
                print(f"  {status} {symbol}: RMSE={rmse:.4f}")
        
        return 0 if successful == total else 1
        
    except Exception as e:
        logger.error(f"Manual retraining failed: {e}")
        return 1

def action_status(config: RetrainingConfig):
    """Show service status"""
    if not AUTOMATION_IMPORTS_AVAILABLE:
        print("❌ Automation components not available")
        return 1
    
    print("📊 Stock AI Automation Status")
    print("=" * 35)
    
    try:
        pipeline = AutomatedRetrainingPipeline(config)
        status = pipeline.get_status()
        
        print(f"Service Running: {status['is_running']}")
        print(f"Monitored Symbols: {status['monitored_symbols']}")
        print(f"Symbols Needing Retraining: {status['symbols_needing_retraining']}")
        print(f"Last Check: {status['last_check']}")
        
        print(f"\nConfiguration:")
        print(f"  Model Type: {config.model_type}")
        print(f"  Feature Level: {config.feature_level}")
        print(f"  Performance Threshold: {config.performance_threshold}")
        print(f"  Schedule: {config.schedule_cron}")
        print(f"  Distributed Training: {config.use_distributed_training}")
        print(f"  Email Notifications: {config.email_notifications}")
        
        if 'performance_summary' in status:
            print(f"\nPerformance Summary:")
            for perf in status['performance_summary']:
                status_icon = "⚠️" if perf['needs_retraining'] else "✅"
                print(f"  {status_icon} {perf['symbol']}: RMSE={perf['rmse']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return 1

def action_config(config: RetrainingConfig, save_path: str = None):
    """Show or save configuration"""
    print("⚙️  Current Configuration")
    print("=" * 30)
    
    config_dict = config.__dict__
    for key, value in config_dict.items():
        if key == 'email_password' and value:
            print(f"{key}: {'*' * len(value)}")  # Mask password
        else:
            print(f"{key}: {value}")
    
    if save_path:
        save_config_to_file(config, save_path)
        print(f"\nConfiguration saved to: {save_path}")
    
    return 0

def main():
    """Main automation service function"""
    args = parse_arguments()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    try:
        # Load configuration
        if args.config_file:
            config = load_config_from_file(args.config_file)
        else:
            config = create_config_from_args(args)
        
        # Save configuration if requested
        if args.save_config:
            save_config_to_file(config, args.save_config)
        
        # Execute action
        if args.action == 'start':
            return action_start(config, daemon=args.daemon)
        elif args.action == 'monitor':
            return action_monitor(config)
        elif args.action == 'retrain':
            return action_retrain(config)
        elif args.action == 'status':
            return action_status(config)
        elif args.action == 'config':
            return action_config(config, args.save_config)
        else:
            print(f"Unknown action: {args.action}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Automation service error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())