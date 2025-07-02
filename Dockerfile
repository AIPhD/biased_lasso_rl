# Use an official Python runtime as a parent image
from python:3.11
# Set the working directory
WORKDIR /home/ls131416/biased_lasso_rl
# Copy the requirements file into the container
COPY requirements.txt .
# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
# Copy the current directory contents into the container at /app
COPY . .
# Run app.py when the container launches
CMD ["python", "run.py"]