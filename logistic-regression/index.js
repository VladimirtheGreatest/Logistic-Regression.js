require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const LogisticRegression = require("./logisic-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["passedemissions"],
    converters: {
      passedemissions: value => {
        return value === 'TRUE' ? 1 : 0;
      }
    }
  }
);

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.5
});

regression.train();
console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.costHistory.reverse(), // newest values are in the front of the array
  xLabel: "Iteration",
  yLabel: "Cost",
});
