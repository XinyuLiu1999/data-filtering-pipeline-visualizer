# Quick Setup Guide

## Your Fixed Access Point

**🌐 http://223.167.203.35:7014**

## Service Management

```bash
# Start the service
./start_service.sh

# Stop the service
./stop_service.sh

# Check status
./status_service.sh
```

## Current Status

- ✅ Application is configured and running
- ✅ Port 7014 is open on server firewall
- ⚠️ **Action Required**: Open port 7014 in your cloud provider's security group

## Next Step: Open Cloud Security Group

Your application is blocked by your cloud provider's firewall. Follow the instructions in [NETWORK_SETUP.md](NETWORK_SETUP.md) to:

1. Log into your cloud provider console
2. Find Security Groups / Firewall Rules
3. Add inbound rule for TCP port 7014
4. Wait 1-2 minutes
5. Access http://223.167.203.35:7014

## Files

- `start_service.sh` - Start the visualizer service
- `stop_service.sh` - Stop the service
- `status_service.sh` - Check if service is running
- `app.py` - Main Flask application
- `NETWORK_SETUP.md` - Instructions to open cloud firewall
- `DEPLOYMENT.md` - Detailed deployment guide

## Troubleshooting

If the service isn't accessible:

1. Check if it's running locally:
   ```bash
   curl http://localhost:7014/health
   # Should return: {"status":"ok"}
   ```

2. Check the logs:
   ```bash
   tail -f logs/app.log
   ```

3. Verify cloud security group allows port 7014 (see NETWORK_SETUP.md)
