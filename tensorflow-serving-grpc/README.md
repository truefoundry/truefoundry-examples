Deploy MobileNet V3 Small with Tensorflow Serving gRPC
---

1. Download the model using `download.sh`
2. Install [`servicefoundry`](https://pypi.org/project/servicefoundry/)
3. Login using e.g. `sfy login --host https://app.truefoundry.com`
4. Deploy to a workspace by passing in the fqn e.g. `python deploy.py --workspace_fqn tfy-ctl-euwe1-devtest:tfy-demo`
5. Visit the deployed service url and copy the endpoint e.g.
   `https://mobilenet-v3-small-tf-tfy-demo.tfy-ctl-euwe1-devtest.devtest.truefoundry.tech`
5. Run the `client.py` in `client/` by passing in the host
   ```shell
   python client.py --host mobilenet-v3-small-tf-tfy-demo.tfy-ctl-euwe1-devtest.devtest.truefoundry.tech
   ```
