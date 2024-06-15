FROM python:3.10.14-slim AS setup
WORKDIR /road_segmentation

FROM setup AS base_req
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Include additional system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0  # Added to provide libgthread

FROM base_req as dev_req
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0  # Ensure all GUI dependencies are covered
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

FROM base_req AS base
COPY . .
RUN pip install .

FROM dev_req as dev
COPY . .
RUN pip install -e ".[dev]"
