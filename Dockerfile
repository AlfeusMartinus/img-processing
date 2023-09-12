# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "img_processing_webgui.py"]
