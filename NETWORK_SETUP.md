# Network Setup Required

## Current Status

✅ **Application Running**: Service is running on port 7014
✅ **Local Access Works**: `http://localhost:7014` responds correctly
✅ **Server Firewall Configured**: iptables rule added for port 7014
❌ **External Access Blocked**: Connection refused from public IP

## Issue

Your server has:
- **Private IP**: 10.233.120.91 (actual server IP)
- **Public IP**: 223.167.203.35 (NAT/gateway IP)

The application is accessible locally but not from the internet. This means **your cloud provider's firewall/security group is blocking port 7014**.

## Solution

You need to configure your cloud provider's security group/firewall to allow inbound traffic on port 7014.

### For Different Cloud Providers:

#### AWS (Amazon Web Services)
1. Go to EC2 Dashboard → Security Groups
2. Find your instance's security group
3. Edit Inbound Rules
4. Add rule:
   - Type: Custom TCP
   - Port: 7014
   - Source: 0.0.0.0/0 (or specific IPs for security)
   - Description: Data Visualizer

#### Alibaba Cloud
1. Go to ECS Console → Network & Security → Security Groups
2. Select your security group
3. Click "Add Rules" → "Inbound"
4. Add rule:
   - Protocol: TCP
   - Port Range: 7014/7014
   - Authorization Object: 0.0.0.0/0
   - Description: Data Visualizer

#### Google Cloud Platform (GCP)
1. Go to VPC Network → Firewall
2. Create firewall rule:
   - Name: allow-visualizer
   - Targets: All instances (or specific tags)
   - Source IP ranges: 0.0.0.0/0
   - Protocols and ports: tcp:7014

#### Azure
1. Go to Virtual Machines → Your VM → Networking
2. Click "Add inbound port rule"
3. Configure:
   - Service: Custom
   - Port ranges: 7014
   - Protocol: TCP
   - Action: Allow
   - Priority: 1000
   - Name: DataVisualizer

#### Tencent Cloud
1. Go to Cloud Virtual Machine → Security Group
2. Select your security group → Inbound Rules
3. Add rule:
   - Type: Custom
   - Protocol: TCP
   - Port: 7014
   - Source: 0.0.0.0/0

## After Adding Security Group Rule

Once you've added the rule in your cloud provider console:

1. Wait 1-2 minutes for the rule to take effect
2. Test access:
   ```bash
   curl http://223.167.203.35:7014/health
   ```
3. If you see `{"status":"ok"}`, it's working!
4. Access in browser: **http://223.167.203.35:7014**

## Current Server Status

```
Service Status: ✅ RUNNING
Process ID: 28333
Port: 7014
Local Access: ✅ http://localhost:7014
External Access: ❌ Blocked by cloud firewall
```

## Quick Check Commands

```bash
# Check if service is running
ps aux | grep "python3 app.py" | grep 7014

# Check if port is listening
ss -tlnp | grep :7014

# Test local access
curl http://localhost:7014/health

# Test external access (after fixing security group)
curl http://223.167.203.35:7014/health
```

## Security Note

When you open port 7014 in the security group:
- The application has no authentication
- Anyone with the URL can access it
- Consider restricting the source IPs in the security group to only allow:
  - Your office IP range
  - Your organization's network
  - Specific trusted IPs

Instead of `0.0.0.0/0` (allow all), use specific IPs like `203.0.113.0/24` (your network range).
