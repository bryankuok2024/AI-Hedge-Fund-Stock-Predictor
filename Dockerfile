# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Assuming your main code is in the 'src' directory and potentially other needed root files
COPY src/ ./src/
# Copy other potential root files if needed by the application (e.g., main.py if called by something)
# COPY main.py .
# COPY backtester.py .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable to tell Streamlit not to open browser on run
ENV STREAMLIT_SERVER_HEADLESS=true

# Run webapp.py when the container launches
# Use --server.address=0.0.0.0 to make it accessible externally
# Use exec form to make CMD the main process (for signals)
CMD ["streamlit", "run", "src/webapp.py", "--server.port=8501", "--server.address=0.0.0.0"] 