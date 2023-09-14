package NeuralNetwork.losses;

public class BinaryCrossentropy implements LossFunction {

    @Override
    public double compute(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }

        double loss = 0.0;
        for (int i = 0; i < actual.length; i++) {
            // Adding a small value to avoid log(0)
            double adjustedPredicted = Math.max(Math.min(predicted[i], 1 - 1e-15), 1e-15);
            loss -= actual[i] * Math.log(adjustedPredicted) + (1 - actual[i]) * Math.log(1 - adjustedPredicted);
        }
        return loss / actual.length;
    }

    @Override
    public double[] derivative(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }

        double[] derivatives = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            derivatives[i] = (predicted[i] - actual[i]) / (predicted[i] * (1 - predicted[i]));
        }
        return derivatives;
    }
}
