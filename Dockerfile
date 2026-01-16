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
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "2", "--timeout", "300", "app:app"]
