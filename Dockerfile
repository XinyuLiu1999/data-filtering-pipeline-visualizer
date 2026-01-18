FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p presets logs

# Expose the port HuggingFace Spaces expects
EXPOSE 7860

# Run with gunicorn
# Use single worker (-w 1) because app stores dataset in memory (global state)
# Multiple workers would cause "No dataset loaded" errors as each worker has separate memory
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "4", "--timeout", "300", "app:app"]
