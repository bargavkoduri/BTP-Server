// import model configuration
const model_config = require("./model_config.json");

// Python shell to run python files
const { PythonShell } = require("python-shell");
const { createServer } = require("http");
const { Server } = require("socket.io");
const ss = require("socket.io-stream");
const fs = require("fs");
const crypto = require("crypto");

// FL parameters
const number_of_required_nodes = 10;
const clients_percentage = 0.2;
const required_percentage = 99.5;
const numberOfRounds = 200;

let started_training = false;
let round = 0;
let best_accuracy = 0;
let accuracies = [];

// Starting the serve in port number 5000
const httpServer = createServer();
let io = new Server(httpServer, {
  maxHttpBufferSize: 1e9,
  pingTimeout: 30000,
});

// Connected clients
let nodes = new Set();
let selected_nodes = new Set();

// Reading and writing to a file
const { ReadFromFile, WriteToFile } = require("./helper");

function getRandomNumber(min, max) {
  let randomNumber = crypto.randomInt(min, max);
  return randomNumber;
}

// function to select random clients
const select_clients = (num, range) => {
  const arr = Array.from(nodes);
  const selected_clients = [];
  const s = new Set();
  while (s.size < num) {
    s.add(getRandomNumber(0, range - 1));
  }
  console.log(`\nSelected clients for round ${round}`);
  for (let ele of s) {
    console.log(arr[ele].id);
    selected_clients.push(arr[ele]);
    selected_nodes.add(arr[ele]);
  }
  return selected_clients;
};

// function to start training
const start_training = () => {
  started_training = true;
  let clients_to_be_selected = clients_percentage * nodes.size;
  clients_to_be_selected = Math.max(clients_to_be_selected, 1);
  round += 1;
  let selected_clients = select_clients(clients_to_be_selected, nodes.size);
  console.log(`Starting round ${round}`);
  selected_clients.forEach((socket) => {
    let stream = ss.createStream();
    ss(socket).emit("start_train", stream);
    fs.createReadStream("model_weights.txt").pipe(stream);
  });
};

const appendToFile = (i, clients_to_be_selected) => {
  ReadFromFile(`weight_updates_${i}.txt`).then((data) => {
    let str;
    if (i == clients_to_be_selected) str = "]";
    else str = ",";
    let newData = JSON.stringify(data) + str;
    fs.appendFile("received_updated_weights.txt", newData, (err) => {
      if (err) console.log("Error appending to file");
      else if (i < clients_to_be_selected)
        appendToFile(i + 1, clients_to_be_selected);
    });
  });
};

// club weights
const clubWeights = () => {
  let clients_to_be_selected = clients_percentage * nodes.size;
  // let clients_to_be_selected = 2;
  WriteToFile("received_updated_weights.txt", "[").then(() => {
    appendToFile(1, clients_to_be_selected);
  });
};

// function for weight update
const update_weights = () => {
  if (selected_nodes.size === 0) {
    console.log("aggregating");
    clubWeights();
    PythonShell.run(
      "aggregator.py",
      {
        scriptPath: "",
        args: [best_accuracy],
      },
      (err, acc) => {
        if (!err) {
          console.log("Finished aggregating");
          console.log(`Accuracy acheived after round ${round} = ${acc}\n`);
          accuracies.push(acc);
          WriteToFile("accuracies.txt", JSON.stringify(accuracies));
          started_training = false;
          weight_updates = [];
          acc = Number(acc);
          if (acc > best_accuracy) best_accuracy = acc;
          if (acc < required_percentage && round < numberOfRounds) {
            start_training();
          }
        } else {
          console.log(err);
        }
      }
    );
  }
};

io.on("connection", (socket) => {
  if (!nodes.has(socket)) {
    // connection information
    console.log("Connected to client : ", socket.id);

    // adding the client to the connected list of clients
    nodes.add(socket);

    // sending the global model on intial connection
    socket.emit("global_model_config", { model_config: model_config });
  }

  // checking for minimum number of clients to start training
  if (nodes.size < number_of_required_nodes) {
    console.log(
      `waiting for ${
        number_of_required_nodes - nodes.size
      } more clients to be connected to start training....`
    );
  } else if (started_training === false) {
    console.log("Required number of clients connected\nstarting training....");
    start_training();
  } else {
    console.log(
      "Traing already started client can participate in the next round\n"
    );
  }

  // weights update
  ss(socket).on("weight_update", (stream) => {
    let fileName = `weight_updates_${selected_nodes.size}.txt`;
    stream.pipe(fs.createWriteStream(fileName));
    stream.on("end", () => {
      console.log(`Model updates received from ${socket.id}`);
      selected_nodes.delete(socket);
      if (selected_nodes.size > 0)
        console.log(`Waiting for ${selected_nodes.size} to send model updates`);
      update_weights();
    });
  });

  // checking for disconnection
  socket.on("disconnect", () => {
    console.log(`Client disconnected with id ${socket.id}`);
    if (started_training === false) {
      nodes.delete(socket);
      selected_nodes.delete(socket);
      console.log(
        `waiting for ${
          number_of_required_nodes - nodes.size
        } more clients to be connected to start training....`
      );
    } else {
      nodes.delete(socket);
      selected_nodes.delete(socket);
    }
  });

  socket.on("error", (err) => {
    console.log(err);
  });
});

httpServer.listen(5000);
