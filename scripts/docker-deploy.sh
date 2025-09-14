#!/bin/bash

# Stock AI Docker Deployment Script
# Professional deployment automation for CV showcase

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
PROJECT_NAME="stock-ai"

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Stock AI Deployment                      ║"
    echo "║              Enterprise Trading System                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "✅ Dependencies check passed"
}

create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "models/saved"
        "data/cache"
        "logs"
        "config"
        "backups"
        "secrets"
        "monitoring"
        "nginx"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

generate_secrets() {
    print_status "Generating secrets..."
    
    # Create secrets directory if it doesn't exist
    mkdir -p secrets
    
    # Generate random passwords if they don't exist
    if [ ! -f secrets/postgres_password.txt ]; then
        openssl rand -base64 32 > secrets/postgres_password.txt
        print_status "Generated PostgreSQL password"
    fi
    
    if [ ! -f secrets/grafana_password.txt ]; then
        openssl rand -base64 16 > secrets/grafana_password.txt
        print_status "Generated Grafana password"
    fi
    
    # Set proper permissions
    chmod 600 secrets/*.txt
}

create_env_file() {
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        
        cat > .env << EOF
# Stock AI Environment Configuration
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
ENVIRONMENT=development

# API Configuration
JWT_SECRET_KEY=$(openssl rand -base64 32)
STOCK_AI_API_KEY=demo-$(openssl rand -hex 16)

# Database Configuration
POSTGRES_DB=stockai
POSTGRES_USER=stockai
POSTGRES_PASSWORD=stockai_password_change_in_production

# Redis Configuration
REDIS_URL=redis://redis:6379

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin_change_in_production

# Feature Flags
ENABLE_MONITORING=true
ENABLE_TRAINING=false
ENABLE_GPU=false
EOF
        
        print_status "Created .env file with default configuration"
        print_warning "Please review and update the .env file before production deployment"
    fi
}

build_images() {
    print_status "Building Docker images..."
    
    if [ "$1" = "prod" ]; then
        docker-compose -f $PROD_COMPOSE_FILE build --no-cache
    else
        docker-compose -f $COMPOSE_FILE build --no-cache
    fi
    
    print_status "✅ Images built successfully"
}

deploy_development() {
    print_status "Deploying development environment..."
    
    # Stop any existing containers
    docker-compose -f $COMPOSE_FILE down --remove-orphans
    
    # Start services
    docker-compose -f $COMPOSE_FILE up -d
    
    print_status "✅ Development environment deployed"
    print_status "Dashboard: http://localhost:8050"
    print_status "API: http://localhost:8000"
    print_status "Redis: localhost:6379"
    print_status "PostgreSQL: localhost:5432"
}

deploy_production() {
    print_status "Deploying production environment..."
    
    # Stop any existing containers
    docker-compose -f $PROD_COMPOSE_FILE down --remove-orphans
    
    # Start services
    docker-compose -f $PROD_COMPOSE_FILE up -d
    
    print_status "✅ Production environment deployed"
    print_status "Load Balancer: http://localhost"
    print_status "Monitoring (Grafana): http://localhost:3000"
    print_status "Metrics (Prometheus): http://localhost:9090"
}

run_training() {
    print_status "Running model training..."
    
    docker-compose -f $COMPOSE_FILE --profile training up stockai-training
    
    print_status "✅ Training completed"
}

show_status() {
    print_status "Current deployment status:"
    echo ""
    docker-compose -f $COMPOSE_FILE ps
}

show_logs() {
    service=${1:-stockai-dashboard}
    print_status "Showing logs for service: $service"
    docker-compose -f $COMPOSE_FILE logs -f $service
}

cleanup() {
    print_status "Cleaning up..."
    
    # Stop and remove containers
    docker-compose -f $COMPOSE_FILE down --remove-orphans
    docker-compose -f $PROD_COMPOSE_FILE down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    print_status "✅ Cleanup completed"
}

show_help() {
    echo "Stock AI Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  dev              Deploy development environment"
    echo "  prod             Deploy production environment"
    echo "  build            Build Docker images"
    echo "  train            Run model training"
    echo "  status           Show deployment status"
    echo "  logs [service]   Show logs for service"
    echo "  stop             Stop all services"
    echo "  clean            Cleanup deployment"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev           # Deploy development environment"
    echo "  $0 prod          # Deploy production environment"
    echo "  $0 logs redis    # Show Redis logs"
    echo "  $0 train         # Run model training"
}

main() {
    print_banner
    
    case "${1:-help}" in
        "dev"|"development")
            check_dependencies
            create_directories
            create_env_file
            build_images
            deploy_development
            ;;
        "prod"|"production")
            check_dependencies
            create_directories
            generate_secrets
            create_env_file
            build_images prod
            deploy_production
            ;;
        "build")
            check_dependencies
            build_images
            ;;
        "train")
            check_dependencies
            run_training
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$2"
            ;;
        "stop")
            docker-compose -f $COMPOSE_FILE down
            docker-compose -f $PROD_COMPOSE_FILE down
            print_status "✅ All services stopped"
            ;;
        "clean")
            cleanup
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"