FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MuJoCo is optional — requires system GL libs not present in slim.
# The app gracefully degrades to MUJOCO_AVAILABLE=False when absent.
RUN pip install --no-cache-dir mujoco==3.2.3 || echo "MuJoCo not available on this platform — running in simulation mode"

COPY api/main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
