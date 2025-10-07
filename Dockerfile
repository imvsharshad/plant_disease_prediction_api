# 1. Start from a stable official Python image.
FROM python:3.9-slim

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy the requirements file into the container.
COPY requirements.txt .

# 4. Install the Python dependencies.
#    --no-cache-dir ensures the image is smaller.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your application files (app.py, model.h5, etc.) into the container.
COPY . .

# 6. Expose the port that your Flask app will run on.
EXPOSE 8080

# 7. Define the command to run your application using gunicorn.
#    This is the command that Hugging Face will execute to start your server.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
