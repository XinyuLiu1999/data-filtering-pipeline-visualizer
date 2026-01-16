# Deployment Guide: Fixed IP Address Access

This guide will help you deploy the Data Filtering Pipeline Visualizer with a fixed IP:port address (e.g., http://101.200.3.202:7014/) accessible from anywhere on the internet.

## Prerequisites

1. **Server with Public IP**: Ensure your server has:
   - A public IP address (check with: `curl ifconfig.me`)
   - Firewall allowing your chosen port (e.g., 7014)

2. **Software Requirements**:
   - Python 3.7+
   - Gunicorn (production WSGI server, recommended but optional)

## Step-by-Step Deployment

### 1. Choose Your Port

Pick a port number for your application (e.g., 7014). Make sure it's:
- Not already in use: `sudo netstat -tulpn | grep :7014`
- Above 1024 (so you don't need root privileges)
- Allowed through your firewall

### 2. Install Python Dependencies

```bash
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

# Install required packages
pip install -r requirements.txt

# Optional: Install Gunicorn for better production performance
pip install gunicorn
```

### 3. Configure Firewall

Allow traffic on your chosen port:

```bash
# For UFW (Ubuntu/Debian)
sudo ufw allow 7014/tcp
sudo ufw status

# For firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=7014/tcp
sudo firewall-cmd --reload

# For iptables
sudo iptables -A INPUT -p tcp --dport 7014 -j ACCEPT
sudo iptables-save
```

### 4. Test the Application

First, test that it works locally:

```bash
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

# Run with custom port
python3 app.py --host 0.0.0.0 --port 7014
```

Visit from your local machine: `http://localhost:7014`
Visit from another machine: `http://YOUR_SERVER_IP:7014`

If it works, press Ctrl+C to stop and proceed to set up the service.

### 5. Set Up Systemd Service (Auto-start on Boot)

```bash
# Edit the service file to use your port (7014)
nano data-visualizer.service
# Change the port in ExecStart line from 5000 to 7014

# Copy service file
sudo cp data-visualizer.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable data-visualizer

# Start the service
sudo systemctl start data-visualizer

# Check status
sudo systemctl status data-visualizer
```

### 6. Get Your Public IP Address

```bash
# Find your public IP
curl ifconfig.me
# or
curl icanhazip.com
```

### 7. Test Your Deployment

Visit your application in a web browser:
```
http://YOUR_PUBLIC_IP:7014
```

For example: `http://101.200.3.202:7014`

You should see the Data Filtering Pipeline Visualizer interface.

## Service Management Commands

```bash
# Start the service
sudo systemctl start data-visualizer

# Stop the service
sudo systemctl stop data-visualizer

# Restart the service
sudo systemctl restart data-visualizer

# Check service status
sudo systemctl status data-visualizer

# View logs
sudo journalctl -u data-visualizer -f

# View nginx logs
sudo tail -f /var/log/nginx/data-visualizer-access.log
sudo tail -f /var/log/nginx/data-visualizer-error.log
```

## Updating the Application

```bash
# Stop the service
sudo systemctl stop data-visualizer

# Pull latest changes or make updates
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer
# Make your changes...

# Restart the service
sudo systemctl start data-visualizer
```

## Alternative Deployment Methods

### Option A: Run Without Systemd (Simple, Manual Start)

If you don't want to set up a systemd service:

```bash
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

# Using Python directly (development)
nohup python3 app.py --host 0.0.0.0 --port 7014 > app.log 2>&1 &

# Using Gunicorn (production, better performance)
nohup gunicorn --workers 4 --bind 0.0.0.0:7014 --timeout 300 app:app > gunicorn.log 2>&1 &
```

To stop the application:
```bash
# Find the process ID
ps aux | grep app.py
# or
ps aux | grep gunicorn

# Kill the process
kill <PID>
```

### Option B: Run in Screen/Tmux Session

```bash
# Using screen
screen -S visualizer
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer
python3 app.py --host 0.0.0.0 --port 7014
# Press Ctrl+A then D to detach

# Reattach later with:
screen -r visualizer

# Using tmux
tmux new -s visualizer
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer
python3 app.py --host 0.0.0.0 --port 7014
# Press Ctrl+B then D to detach

# Reattach later with:
tmux attach -t visualizer
```

## Security Considerations

**⚠️ Important**: When exposing services on public IP addresses:

1. **No Authentication**: The current app has no authentication. Anyone who knows your IP:port can access it. Consider:
   - Adding IP whitelisting in the firewall
   - Implementing basic authentication
   - Using VPN for access
   - Keeping the port number private

2. **HTTP (not HTTPS)**: Traffic is unencrypted. Data sent over the network can be intercepted.
   - For sensitive data, use a VPN or SSH tunnel
   - Or follow the full nginx+SSL setup (see nginx.conf file)

3. **File Access**: Ensure the app only accesses authorized dataset paths

4. **Firewall Rules**: Only allow access from trusted IP ranges if possible:
```bash
# Example: Only allow access from specific IP range
sudo ufw allow from 192.168.1.0/24 to any port 7014

# Example: Allow from specific IP
sudo ufw allow from 101.200.5.100 to any port 7014
```

## Troubleshooting

### Service won't start
```bash
# Check logs
sudo journalctl -u data-visualizer -n 50

# Test running manually
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer
python3 app.py
```

### Cannot access from internet
- Check firewall: `sudo ufw status`
- Verify public IP: `curl ifconfig.me`
- Test if port is open: `sudo netstat -tulpn | grep :7014`
- Check if service is running: `sudo systemctl status data-visualizer`
- Check cloud provider security groups (AWS, Azure, GCP, etc.)
- Check router port forwarding if behind NAT

### Port already in use
```bash
# Find what's using the port
sudo lsof -i :7014
# or
sudo netstat -tulpn | grep :7014

# Kill the process if needed
sudo kill <PID>
```

### Permission denied on port
If you get "Permission denied" when trying to use a port below 1024, either:
- Use a port above 1024 (recommended)
- Or run with sudo (not recommended for security)

## Production Recommendations

1. **Use Gunicorn with multiple workers** (already configured in systemd service)
2. **Regular backups** of presets and configuration
3. **Log rotation** to prevent disk space issues
4. **Update dependencies** regularly for security patches
5. **Consider adding authentication** for security

## Your Fixed Access Point

Once deployed, your application will be accessible at:

**🌐 http://YOUR_PUBLIC_IP:7014**

For example: **http://101.200.3.202:7014**

This address is:
- ✅ Fixed and permanent (as long as your IP doesn't change)
- ✅ Accessible from anywhere on the internet
- ✅ No domain registration required
- ✅ Easy to share with colleagues

## Quick Start Summary

The fastest way to get started:

```bash
# 1. Go to your project directory
cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

# 2. Install dependencies
pip install -r requirements.txt gunicorn

# 3. Open firewall
sudo ufw allow 7014/tcp

# 4. Edit and install systemd service
nano data-visualizer.service  # Change port to 7014
sudo cp data-visualizer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable data-visualizer
sudo systemctl start data-visualizer

# 5. Get your public IP
curl ifconfig.me

# 6. Access at http://YOUR_IP:7014
```

That's it! Your visualizer is now accessible at a fixed address.
