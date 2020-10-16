require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");

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



console.log(labels);