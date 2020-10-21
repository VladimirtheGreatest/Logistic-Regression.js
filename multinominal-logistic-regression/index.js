require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const LogisticRegression = require("./logisic-regression");
const plot = require("node-remote-plot");
const _ = require('lodash');

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["mpg"],
    //converter function helps us to convert values from mpg to array [1, 0, 0] we need to flatten the array later to get rid of extra nesting
    converters: {
      mpg: value => {
        const mpg = parseFloat(value);

        if (mpg < 15) {
          return [1, 0, 0];
        } else if (mpg < 30){
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      }
    }
  }
);

//console.log(_.flatMap(labels));

const regression = new LogisticRegression(features, _.flatMap(labels), {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10
});

 regression.train();

console.log(regression.test(testFeatures, _.flatMap(testLabels)));
