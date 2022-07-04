# Deploying an Gradio webapp for MNIST digit recognition

1. Before moving forward, make sure you have:
   1. Installed servicefoundry library and created a TrueFoundry workspace. If not, you can quickly do it by [following the instructions here](https://docs.truefoundry.com/documentation/deploy-model/quick-start/install-and-workspace).
   2. Trained the model and copied the run id by running the [training notebook](../train.ipynb). You can also get the run id from the [TrueFoundry dashboard](https://app.truefoundry.com/mlfoundry).
2. In the [servicefoundry.yaml](./servicefoundry.yaml) file, replace the run id with your own run id and the workspace FQN with your workspace FQN. You can copy your workspace FQN from the [workspaces dashboard](https://app.truefoundry.com/workspace).
3. Now simply run, `sfy deploy` from this folder to deploy your webapp.