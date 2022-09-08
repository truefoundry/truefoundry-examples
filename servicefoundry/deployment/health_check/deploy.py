import logging

from servicefoundry import (
    Build,
    PythonBuild,
    Service,
    HttpProbe,
    HealthProbe,
)

logging.basicConfig(level=logging.INFO)
service = Service(
    name="svc-health",
    image=Build(
        build_spec=PythonBuild(
            command="uvicorn main:app --port 8000 --host 0.0.0.0",
            pip_packages=["fastapi==0.81.0", "uvicorn==0.18.3"],
        ),
    ),
    ports=[{"port": 8000}],
    liveness_probe=HealthProbe(
        config=HttpProbe(path="/livez", port=8000),
        initial_delay_seconds=0,
        period_seconds=10,
        timeout_seconds=1,
        success_threshold=1,
        failure_threshold=3,
    ),
    readiness_probe=HealthProbe(
        config=HttpProbe(path="/readyz", port=8000),
        period_seconds=5,
    ),
)
# service.deploy(workspace_fqn="YOUR_WORKSPACE_FQN")
service.deploy(workspace_fqn="v1:local:my-ws-2")
