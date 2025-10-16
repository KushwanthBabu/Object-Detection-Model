# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Tell Fly.io that the app listens on port 8080
EXPOSE 8080

# The command that will be run to start your web server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]