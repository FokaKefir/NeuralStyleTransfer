# Use an official Nginx image as the base image
FROM nginx:latest

# Install necessary packages for PostgreSQL
RUN apt-get update && apt-get install -y postgresql-client postgresql

# Install Python, PyTorch, FastAPI, and Uvicorn
RUN apt-get install -y python3.10 python3-pip python3-venv
RUN python3 -m venv /app/env
RUN /app/env/bin/pip install torch torchvision torchaudio fastapi uvicorn


# Set up a directory for the application code
WORKDIR /app

# Expose ports
EXPOSE 80
EXPOSE 8000

# Start Nginx and PostgreSQL services
#CMD service nginx start && service postgresql start
CMD ["nginx", "-g", "daemon off;"]