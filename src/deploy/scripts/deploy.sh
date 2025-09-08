#!/bin/bash

# Stock AI Production Deployment Script

set -e

echo "🚀 Starting Stock AI Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_KEY=${API_KEY:-$(openssl rand -hex 32)}
DOCKER_COMPOSE_FILE="docker/docker-compose.yml"
ENV_FILE=".env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > $ENV_FILE << EOF
# Stock AI Production Environment Configuration
API_KEY=$API_KEY
REDIS_URL=redis://redis:6379
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
RATE_LIMIT_CALLS=1000
RATE_LIMIT_PERIOD=3600
CACHE_TTL=300
MAX_PREDICTION_DAYS=90
DEFAULT_PREDICTION_DAYS=30
MODEL_CACHE_SIZE=10
CORS_ORIGINS=*

# Database (if needed in future)
# DATABASE_URL=postgresql://user:pass@localhost/stockai

# Monitoring (if needed)
# SENTRY_DSN=your-sentry-dsn-here
# PROMETHEUS_ENABLED=true
EOF
    
    print_success "Environment file created: $ENV_FILE"
    print_warning "API Key generated: $API_KEY"
    print_warning "Please save this API key securely!"
}

# Setup directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p models/saved
    mkdir -p plots
    mkdir -p logs
    mkdir -p docker/ssl
    
    # Set proper permissions
    chmod 755 models plots logs
    
    print_success "Directories created"
}

# Build and start services
deploy_services() {
    print_status "Building and deploying services..."
    
    # Stop existing services
    docker-compose -f $DOCKER_COMPOSE_FILE down 2>/dev/null || true
    
    # Build and start services
    docker-compose -f $DOCKER_COMPOSE_FILE build --no-cache
    docker-compose -f $DOCKER_COMPOSE_FILE up -d
    
    print_success "Services deployed"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Wait for services to start
    sleep 10
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "API is healthy and responding"
            break
        fi
        
        if [ $i -eq 30 ]; then
            print_error "API health check failed"
            docker-compose -f $DOCKER_COMPOSE_FILE logs stock-ai-api
            exit 1
        fi
        
        print_status "Waiting for API to start... (attempt $i/30)"
        sleep 2
    done
    
    # Check Redis
    if docker-compose -f $DOCKER_COMPOSE_FILE exec -T redis redis-cli ping | grep -q "PONG"; then
        print_success "Redis is healthy"
    else
        print_warning "Redis health check failed"
    fi
}

# Display deployment information
show_deployment_info() {
    echo
    echo "🎉 Stock AI API Successfully Deployed!"
    echo "======================================"
    echo
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "🔧 Health Check: http://localhost:8000/health"
    echo "📊 Status: http://localhost:8000/status"
    echo "🔑 API Key: $API_KEY"
    echo
    echo "📋 Available Endpoints:"
    echo "  POST /predict - Stock price predictions"
    echo "  POST /portfolio/optimize - Portfolio optimization"
    echo "  POST /regime/analyze - Market regime analysis"
    echo "  GET /models/list - List available models"
    echo
    echo "🐳 Docker Services:"
    echo "  stock-ai-api: Main API service"
    echo "  redis: Caching layer"
    echo "  nginx: Reverse proxy and load balancer"
    echo
    echo "📝 Management Commands:"
    echo "  docker-compose -f $DOCKER_COMPOSE_FILE logs -f     # View logs"
    echo "  docker-compose -f $DOCKER_COMPOSE_FILE ps          # Check status"
    echo "  docker-compose -f $DOCKER_COMPOSE_FILE down        # Stop services"
    echo "  docker-compose -f $DOCKER_COMPOSE_FILE up -d       # Start services"
    echo
    echo "🔒 Security Notes:"
    echo "  - Change the default API key in production"
    echo "  - Configure SSL certificates for HTTPS"
    echo "  - Set up proper firewall rules"
    echo "  - Configure monitoring and logging"
    echo
}

# Cleanup function
cleanup() {
    print_warning "Deployment interrupted. Cleaning up..."
    docker-compose -f $DOCKER_COMPOSE_FILE down 2>/dev/null || true
    exit 1
}

# Set trap for cleanup
trap cleanup INT TERM

# Main deployment flow
main() {
    print_status "Starting Stock AI Production Deployment"
    
    check_docker
    create_env_file
    setup_directories
    deploy_services
    health_check
    show_deployment_info
    
    print_success "Deployment completed successfully! 🚀"
}

# Run main function
main "$@"