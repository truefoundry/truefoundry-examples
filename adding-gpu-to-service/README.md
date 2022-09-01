### Adding GPUs to your servicefoundry service

You can add GPUs to services by simply adding a limit in the servicefoundry.yaml file.

```
name: gpu-test
components:
  - name: gpu-test
    type: service
  ...
    resources:
      gpu_limit: 1
  ...
```

Check the [`service/`](./service/) directory for a minimal example and how to deploy it
