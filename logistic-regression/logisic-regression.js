const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    //if we dont provide learning rate in options the default rate will be 0.1
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    //initial guesses, M and B previously
    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    //mathmul matrix multiplication
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose() // RESHAPING TENSOR so we can match the shape of differences
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate)); //this.m = this.m - mSlope * this.options.learningRate;
  }

  train() {
    //batch gradient descent        rows/batchsize == how many times we iterate
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    ); // in case we have soem leftover odd number

    for (let index = 0; index < this.options.iterations; index++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const batchSize = this.options.batchSize;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
    }
  }
  
  test(testFeatures,testLabels){
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    //coefficient of determination  R2 = 1 - total sum of squares / sum of squares of residuals  check notes, aka gauging accuracy of our prediction

    //sum of squares of residuals
    const res = testLabels.sub(predictions)
    .pow(2)
    .sum() // we dont have to provide axis for this
    .get()
    //total sum of squares
    const tot = testLabels
    .sub(testLabels.mean())
    .pow(2)
    .sum()
    .get();

    return 1 - res / tot;
  }

  predict(observations){
    return this.processFeatures(observations).matMul(this.weights).sigmoid();
  }

  processFeatures(features) {
    //generates an extra column so we can use the matrix multiplication
    //ones([shape]) shape = features row, one column,    1 for concatenation axis
    features = tf.tensor(features);

    //we have to reapply mean and variance for our test set if it is second time
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
      //we use our helper function for the first case
    } else {
      features = this.standardize(features);
    }
    //column of ones needs to come after standardization so we do not change the ones values
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }
  //vectorized solution
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0]) //number of observations
      .get();

    //put the most recet mse in the beginning of the array
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }
    //if the value of mse goes up we are overshooting and getting incorrect values we need to decrease our learning rate
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate = this.options.learningRate / 2;
    } else {
      this.options.learningRate *= 1.05; //increase the learning by 5% if the MSE error goes down and we are getting closer to the optimal value
    }
  }
}

module.exports = LogisticRegression;
