# Annotation Service Troubleshooting Guide

## Issue: `/annotation/` returns 404

This document outlines the fixes applied to resolve the broken connection to `tools/annotation` service.

## Root Causes Identified

1. **Nginx Configuration Issues**: Missing `X-Script-Name` headers and inconsistent upstream references
2. **Error Handling**: Poor error reporting made debugging difficult
3. **Health Checks**: No easy way to verify service connectivity

## Fixes Applied

### 1. Enhanced Nginx Configuration (`nginx-compose.conf`)

- **Added `X-Script-Name` headers** to all proxy locations for proper Flask URL generation
- **Improved error handling** with fallback locations and better error messages
- **Added debug headers** to help troubleshoot routing issues
- **Added annotation health check** endpoint at `/annotation/health`

### 2. Better Error Reporting

```nginx
# Fallback location for annotation service errors
location @annotation_fallback {
    add_header Content-Type text/plain;
    return 503 "Annotation service temporarily unavailable. Please check that the annotation-app container is running.";
}
```

### 3. Debug Tools Created

- **`test_annotation.py`**: Python script to test service connectivity
- **`docker-compose.debug.yml`**: Debug configuration with enhanced logging
- **`start_debug.sh`**: Automated startup and testing script

## Testing the Fix

### Quick Test
```bash
# Start services
cd /workspace/dashboard
./start_debug.sh

# Or manually:
docker compose up -d
python3 test_annotation.py localhost
```

### Manual Verification
1. **Dashboard**: http://localhost:8080 → Should show dashboard
2. **Annotation Tool**: http://localhost:8080/annotation/ → Should show annotation interface
3. **Health Check**: http://localhost:8080/annotation/health → Should return session data
4. **Direct Flask**: http://localhost:5000 → Should work if annotation-app is running

## Common Issues and Solutions

### 1. Service Not Starting
```bash
# Check container status
docker compose ps

# Check logs
docker compose logs annotation-app
docker compose logs nginx
```

### 2. Network Connectivity Issues
```bash
# Test internal network
docker compose exec nginx ping annotation-app

# Check port bindings
docker compose port nginx 80
docker compose port annotation-app 5000
```

### 3. Static Files Not Loading
- Verify nginx proxy configuration for `/annotation/static/`
- Check Flask `static_url_path` configuration in `app.py`
- Ensure static files exist in `tools/annotation/static/`

### 4. API Endpoints Failing
- Check that API routes are properly registered in Flask
- Verify nginx proxy rules for `/annotation/api/` paths
- Test direct Flask API access: http://localhost:5000/api/sessions

## Configuration Summary

### Docker Compose Services
- **annotation-app**: Flask service on port 5000 (internal network)
- **nginx**: Reverse proxy on port 8080 (external access)

### Nginx Routing
- `/` → Dashboard static files
- `/annotation/` → Flask app root (`annotation-app:5000/`)
- `/annotation/api/` → Flask API (`annotation-app:5000/api/`)
- `/annotation/static/` → Flask static files
- `/health` → Nginx health check
- `/annotation/health` → Flask app health check

### Flask Configuration
- **Application Root**: `/annotation` (for reverse proxy)
- **Static URL Path**: `/annotation/static`
- **Routes**: `/` and `/annotation/` both serve the main interface

## Debugging Commands

```bash
# View nginx access logs
docker compose exec nginx tail -f /var/log/nginx/access.log

# View nginx error logs
docker compose exec nginx tail -f /var/log/nginx/error.log

# Test nginx configuration
docker compose exec nginx nginx -t

# Restart services
docker compose restart

# View Flask logs
docker compose logs -f annotation-app
```

## Next Steps

If the issue persists after applying these fixes:

1. Check Docker network connectivity
2. Verify that both services are healthy
3. Test direct Flask access on port 5000
4. Review nginx and Flask logs for specific errors
5. Use the provided test script to identify the failing component

The configuration should now properly route requests from `/annotation/` to the Flask application while maintaining proper static file serving and API functionality.