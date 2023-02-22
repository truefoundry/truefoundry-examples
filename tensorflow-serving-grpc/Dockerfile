FROM tensorflow/serving:2.11.0
COPY models/ /mnt/models
COPY start_server.sh /app/start_server.sh
ENTRYPOINT ["/app/start_server.sh"]
