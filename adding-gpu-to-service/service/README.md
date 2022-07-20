### Minimal Pytorch example using GPUs
 
To deploy this minimal example,

1. Signup on [TrueFoundry](https://app.truefoundry.com/)

2. Install servicefoundry CLI and login

   ```shell
   pip install servicefoundry
   sfy login
   ```

2. Create (or use and existing) a workspace from the
   [dashboard](https://app.truefoundry.com/workspace). You can also use
   cli

   ```shell
   sfy create workspace <workspace-name>
   ```

   Once you have created the workspace copy the `fqn` field

3. Clone this repository and change into the service directory
   
   ```shell
   git clone https://github.com/truefoundry/truefoundry-examples
   cd truefoundry-examples/adding-gpu-to-service/service/
   ```

4. Run `sfy deploy`

   ```shell
   sfy deploy --workspace <your-workspace-fqn>
   ```

   > You can also choose to put workspace fqn directly in the [`servicefoundry.yaml`](./servicefoundry.yaml#L13) and then deploy with just `sfy deploy`
