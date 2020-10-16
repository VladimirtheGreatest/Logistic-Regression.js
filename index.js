require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(), //we have to reverse iteration
  xLabel: "Iteration",
  yLabel: "Mean Squared Error",
});

console.log("Coefficient of determination is", r2);
//negative value means bad result, we need to improve accuracy of our analysis

//["horsepower", "weight", "displacement"]
regression.predict([
  [46, 1.75, 307]
]).print();
//result mile per gallon prediction
