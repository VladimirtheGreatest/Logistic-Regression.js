//old implementation
   export function gradientDescentOld() {
    const currentGuessesForMPG = this.features.map((row) => {
      //inner array calculation or features * weights
      return this.m * row[0] + this.b;
    });

    //formula   sum of ((m * feature) + b) - label(actual value)
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return guess - this.labels[i][0];   //inner array - labels
        })
      ) *
        2) /
      this.features.length; //  * derivative / number of observation can be either length of features or labels or N

    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })
      ) *
        2) /
      this.features.length;

      this.m = this.m - mSlope * this.options.learningRate;
      this.b = this.b - bSlope * this.options.learningRate;
    }
  
