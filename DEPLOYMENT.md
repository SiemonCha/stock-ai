# Stock AI Deployment Guide 🚀

Professional deployment documentation for the Stock AI trading system.

## Quick Start

### 🐳 One-Command Deployment
```bash
# Development environment
./scripts/docker-deploy.sh dev

# Production environment  
./scripts/docker-deploy.sh prod
```

### 📊 Access Points
- **Dashboard**: http://localhost:8050
- **API Docs**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)

## Deployment Options

### Development Environment
```bash
# Clone and deploy
git clone https://github.com/yourusername/stock-ai
cd stock-ai
./scripts/docker-deploy.sh dev
```

**Services Included:**
- Interactive Dashboard (Port 8050)
- REST API (Port 8000) 
- Redis Cache (Port 6379)
- PostgreSQL Database (Port 5432)

### Production Environment
```bash
# Production with load balancing
./scripts/docker-deploy.sh prod
```

**Additional Features:**
- Nginx Load Balancer
- Multiple Dashboard Instances
- Prometheus Monitoring
- Grafana Visualization
- Log Management (Loki)

## Architecture

### Container Structure
```
┌─────────────────┐    ┌─────────────────┐
│  Nginx (LB)     │    │  Dashboard-1    │
│  Port: 80       │───▶│  Port: 8050     │
└─────────────────┘    └─────────────────┘
                            │
                       ┌─────────────────┐
                       │  Dashboard-2    │
                       │  Port: 8050     │
                       └─────────────────┘
                            │
         ┌─────────────────────────────────────┐
         │              Data Layer             │
         │  ┌─────────┐ ┌──────────┐ ┌──────┐ │
         │  │ Redis   │ │PostgreSQL│ │ Loki │ │
         │  └─────────┘ └──────────┘ └──────┘ │
         └─────────────────────────────────────┘
```

## Configuration

### Environment Variables
```bash
# Core Configuration
ENVIRONMENT=production
REDIS_URL=redis://redis:6379
API_HOST=0.0.0.0
API_PORT=8000

# Security
JWT_SECRET_KEY=your-secret-key
STOCK_AI_API_KEY=your-api-key

# Database
POSTGRES_DB=stockai
POSTGRES_USER=stockai
POSTGRES_PASSWORD=secure-password

# Features
ENABLE_MONITORING=true
ENABLE_TRAINING=false
ENABLE_GPU=false
```

### Docker Compose Profiles

**Development Stack:**
```yaml
services:
  - stockai-dashboard
  - redis  
  - postgres
```

**Production Stack:**
```yaml
services:
  - nginx (load balancer)
  - stockai-dashboard-1
  - stockai-dashboard-2
  - redis
  - postgres
  - prometheus
  - grafana
  - loki
```

## Commands Reference

### Deployment Commands
```bash
# Deploy development
./scripts/docker-deploy.sh dev

# Deploy production
./scripts/docker-deploy.sh prod

# Build only
./scripts/docker-deploy.sh build

# Show status
./scripts/docker-deploy.sh status

# View logs
./scripts/docker-deploy.sh logs [service]

# Stop services
./scripts/docker-deploy.sh stop

# Cleanup
./scripts/docker-deploy.sh clean
```

### Docker Compose Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f stockai-dashboard

# Scale dashboard
docker-compose up -d --scale stockai-dashboard=3

# Restart service
docker-compose restart stockai-dashboard

# Update and redeploy
docker-compose pull && docker-compose up -d
```

### Training Commands
```bash
# Run model training
./scripts/docker-deploy.sh train

# Custom training
docker-compose --profile training run stockai-training \
  python train_models.py --symbols AAPL,GOOGL --epochs 50
```

## Monitoring & Observability

### Grafana Dashboards
- **System Metrics**: CPU, memory, network usage
- **Application Metrics**: API latency, prediction accuracy
- **Business Metrics**: Trading performance, portfolio value

### Prometheus Metrics
- `stockai_predictions_total`: Total predictions made
- `stockai_api_requests_duration`: API response time
- `stockai_model_accuracy`: Current model accuracy
- `stockai_trading_pnl`: Profit/loss metrics

### Log Management
- **Application Logs**: `/app/logs/`
- **System Logs**: Collected by Loki
- **Access Logs**: Nginx request logs

## Security

### Network Security
- Rate limiting (API: 10 req/s, Dashboard: 30 req/s)
- CORS protection
- Security headers (XSS, CSRF protection)
- Internal network isolation

### Data Security
- Non-root container users
- Secrets management
- Environment variable encryption
- Secure database passwords

## Performance Optimization

### Resource Allocation
```yaml
# Recommended resources
stockai-dashboard:
  cpu: "1.0"
  memory: "2GB"
  
redis:
  cpu: "0.5"
  memory: "1GB"
  
postgres:
  cpu: "1.0" 
  memory: "2GB"
```

### Scaling Strategies
```bash
# Horizontal scaling
docker-compose up -d --scale stockai-dashboard=5

# Load testing
ab -n 1000 -c 10 http://localhost/api/predictions/AAPL
```

## Troubleshooting

### Common Issues

**Dashboard not accessible:**
```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs stockai-dashboard

# Verify ports
netstat -tlnp | grep 8050
```

**Database connection errors:**
```bash
# Check PostgreSQL
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U stockai -d stockai
```

**Redis connection issues:**
```bash
# Check Redis
docker-compose logs redis

# Test Redis
docker-compose exec redis redis-cli ping
```

### Health Checks
```bash
# Dashboard health
curl http://localhost:8050/health

# API health  
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready
```

## Backup & Recovery

### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U stockai stockai > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U stockai stockai < backup.sql
```

### Model Backup
```bash
# Backup models
docker-compose exec stockai-dashboard tar -czf /tmp/models.tar.gz /app/models

# Copy from container
docker cp stockai-dashboard:/tmp/models.tar.gz ./models-backup.tar.gz
```

## Production Checklist

### Before Deployment
- [ ] Update environment variables
- [ ] Generate secure secrets
- [ ] Configure SSL certificates
- [ ] Set up monitoring alerts
- [ ] Test backup procedures
- [ ] Review resource limits
- [ ] Configure log rotation

### Post Deployment
- [ ] Verify all services are running
- [ ] Test dashboard accessibility
- [ ] Validate API responses
- [ ] Check monitoring dashboards
- [ ] Verify database connectivity
- [ ] Test backup/restore procedures
- [ ] Monitor resource usage

## Support

### Documentation
- **API Docs**: http://localhost:8000/docs
- **Dashboard Help**: Available in UI
- **System Status**: http://localhost:8050/status

### Logging Locations
- **Container Logs**: `docker-compose logs [service]`
- **Application Logs**: `./logs/`
- **Access Logs**: `./nginx/logs/`

---

*For enterprise support and custom deployment options, contact the development team.*