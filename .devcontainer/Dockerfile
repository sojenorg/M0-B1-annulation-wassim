FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt repo whose GPG key has expired, which breaks
# `apt-get update` (and thus the docker-in-docker feature). Yarn isn't needed here.
RUN rm -f /etc/apt/sources.list.d/yarn.list
