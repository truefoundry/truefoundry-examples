Deploy Python gRPC Server with Truefoundry
---

Example taken and modified from [gRPC Python quickstart](https://grpc.io/docs/languages/python/quickstart)

1. Install [`servicefoundry`](https://pypi.org/project/servicefoundry/)
2. Login using e.g. `sfy login --host https://app.truefoundry.com`
3. Deploy to a workspace by passing in the fqn e.g. `python deploy.py --workspace_fqn tfy-ctl-euwe1-devtest:tfy-demo`
4. Visit the deployed service url and copy the endpoint e.g.
   `https://grpc-py-helloworld-tfy-demo.tfy-ctl-euwe1-devtest.devtest.truefoundry.tech`
5. Run the `greeter_client_with_options.py` by passing in the host
   ```shell
   python greeter_client_with_options.py --host grpc-py-helloworld-tfy-demo.tfy-ctl-euwe1-devtest.devtest.truefoundry.tech --name Truefoundry
   ```
