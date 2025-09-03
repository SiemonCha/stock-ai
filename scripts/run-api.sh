#!/bin/bash
# Stock AI API startup script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Stock AI API...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️ .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file. Please update it with your configuration.${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p models/saved data/cache plots logs
mkdir -p models/saved/.gitkeep data/cache/.gitkeep plots/.gitkeep logs/.gitkeep

# Build and start services
echo -e "${BLUE}🔨 Building Docker images...${NC}"
docker-compose build

echo -e "${BLUE}🚢 Starting services...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check health
echo -e "${BLUE}🔍 Checking service health...${NC}"

# Check API health
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ Stock AI API is running at http://localhost:8000${NC}"
    echo -e "${GREEN}📚 API Documentation: http://localhost:8000/docs${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    echo -e "${YELLOW}📋 Checking logs:${NC}"
    docker-compose logs stockai-api
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not responding${NC}"
fi

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U stockai > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL is running${NC}"
else
    echo -e "${RED}❌ PostgreSQL is not responding${NC}"
fi

echo -e "${BLUE}📊 Service Status:${NC}"
docker-compose ps

echo -e "${GREEN}🎉 Stock AI is ready!${NC}"
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  View logs: ${YELLOW}docker-compose logs -f stockai-api${NC}"
echo -e "  Stop services: ${YELLOW}docker-compose down${NC}"
echo -e "  Restart services: ${YELLOW}docker-compose restart${NC}"
echo -e "  Access database: ${YELLOW}docker-compose exec postgres psql -U stockai stockai${NC}"