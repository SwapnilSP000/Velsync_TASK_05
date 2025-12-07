# Task 05 - ML App Dockerfile

# Use lightweight Python image
FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port
EXPOSE 5000

# Run service with gunicorn (production server)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
