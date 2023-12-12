# FL-Server
The Federated Learning Server is a centralized component of the federated learning system designed to manage the training process of distributed machine learning models across multiple devices or nodes. 
This server acts as the coordinating entity that facilitates communication, aggregation, and model updates between participating devices without compromising data privacy.
<br/>
You can find the link to the client code <a href="https://github.com/bargavkoduri/BTP-Client">here</a>

# Set-Up
To run the server make sure you have node, python and tensorflow installed on your system.
<br/>
To install other dependencies type
<br/>
``` npm i ```

# Usage
1) Run the below command to initialiaze the model : <br/>
   ```python init_model.py```
<br/>You can tweek the neural network by editing the init_model-name.py file.<br/>
All the models above accept 224x224x3 images for 12 class classification.

3) Run ```node server.js``` to start the server <br/>
<br/>
You can change the number of clients required to start the process and client percentage and accuracy at which the process stops in server.js file.
You can stop it at any time by killing the process. It stores best model (model with highest accuracy acheived till now).
