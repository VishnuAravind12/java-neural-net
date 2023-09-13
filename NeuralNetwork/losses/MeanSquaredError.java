package NeuralNetwork.losses;

public class MeanSquaredError {

    public double compute(double[] actual, double[] predicted) {
        //Error Handling for if the predicted set of values does not correspond in length with the true set of values.
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have the same length.");
        }

        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) {
            sum += Math.pow(actual[i] - predicted[i], 2);
        }
        double mse = sum / actual.length;
        
        return mse;
    }

    public double[] derivative(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }

        double[] derivatives = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            derivatives[i] = 2 * (predicted[i] - actual[i]);
        }
        return derivatives;
    }
}