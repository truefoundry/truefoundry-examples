#!/bin/bash

# Make sure you have logged in using `docker login`
# Replace `<YOUR_DOCKERHUB_USERNAME>` with the actual value.
docker build . -t <YOUR_DOCKERHUB_USERNAME>/tf-job-docker-image:latest
docker push <YOUR_DOCKERHUB_USERNAME>/tf-job-docker-image:latest
