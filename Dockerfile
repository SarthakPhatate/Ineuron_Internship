FROM python:3.6

# Add sample application
COPY * /tmp/

EXPOSE 8000

# Run it
RUN pip install -r /tmp/requirements.txt
ENTRYPOINT ["python", "/tmp/application.py"]
