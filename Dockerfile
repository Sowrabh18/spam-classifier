# ─────────────────────────────────────────────────────
# Dockerfile
#
# Builds a container image for the spam classifier API.
# Base: Python 3.10 slim (lightweight Linux + Python)
# Exposes: port 8000 (FastAPI via uvicorn)
# ─────────────────────────────────────────────────────

# ── Base image ─────────────────────────────────────────
# We start from an official Python image on Docker Hub
# python:3.10-slim is a lightweight version of
# Debian Linux with Python 3.10 already installed
# 'slim' means unnecessary system packages are removed
# making the image smaller and faster to download
FROM python:3.10-slim

# ── Working directory ──────────────────────────────────
# Creates a folder called /app inside the container
# and makes it the current working directory
# Every command after this runs inside /app
# Think of it as cd /app but it also creates the folder
WORKDIR /app

# ── Copy requirements first ────────────────────────────
# We copy requirements.txt BEFORE copying the rest
# of the code. This is a Docker best practice.
#
# Why? Docker builds in layers. Each instruction is
# one layer. Layers are cached — if nothing changed
# in that layer Docker reuses the cached version
# instead of rebuilding it.
#
# requirements.txt changes rarely. Code changes often.
# By copying requirements first and installing libraries
# before copying code — the pip install layer gets
# cached. Next time you build Docker skips pip install
# entirely if requirements.txt hasn't changed.
# This makes rebuilds much faster.
COPY requirements.txt .

# ── Install libraries ──────────────────────────────────
# Runs pip install inside the container
# --no-cache-dir tells pip not to store downloaded
# packages in a cache folder — keeps image size small
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of the project ───────────────────────
# Now copy everything else into /app
# This happens after pip install so code changes
# don't invalidate the pip install cache layer
COPY . .

# ── Expose port ────────────────────────────────────────
# Tells Docker this container will listen on port 8000
# This is documentation — it doesn't actually open
# the port. The actual port mapping happens when
# you run the container with docker run -p 8000:8000
EXPOSE 8000

# ── Start command ──────────────────────────────────────
# This runs when the container starts
# uvicorn main:app starts our FastAPI app
# --host 0.0.0.0 means accept connections from
# anywhere — not just localhost
# Without this the API would only be accessible
# from inside the container itself — useless
# --port 8000 matches the EXPOSE above
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]