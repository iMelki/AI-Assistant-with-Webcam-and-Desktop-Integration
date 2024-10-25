# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Use a different mirror for Debian packages
# RUN sed -i 's|http://deb.debian.org/debian|http://ftp.de.debian.org/debian|g' /etc/apt/sources.list

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "assistant.py"]

# Define environment variable for the API keys
# ENV OPENAI_API_KEY=your_openai_api_key_here
# ENV GOOGLE_API_KEY=your_google_api_key_here
