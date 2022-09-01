# Deploying an Gradio webapp for MNIST digit recognition

1. Before moving forward, make sure you have:
   1. Installed servicefoundry library and created a TrueFoundry workspace. If not, you can quickly do it by [following the instructions here](https://docs.truefoundry.com/documentation/deploy-model/quick-start/install-and-workspace).
   2. Trained the model and copied the run id by running the [training notebook](../train.ipynb). You can also get the run id from the [TrueFoundry dashboard](https://app.truefoundry.com/mlfoundry) after training.
2. In the `servicefoundry.yaml` file:
   1.  Replace the [run id](./servicefoundry.yaml#L16) with your own run id
3. Now simply run, `sfy deploy --workspace-fqn <your-workspace-fqn>` in this folder to deploy your service. Copy your workspace FQN from the [workspaces dashboard](https://app.truefoundry.com/workspace).
4. Access your service from the [dashboard](https://app.truefoundry.com/applications).
