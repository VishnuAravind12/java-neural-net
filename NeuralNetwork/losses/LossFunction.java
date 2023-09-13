package NeuralNetwork.losses;

public interface LossFunction {

    /**
     * Computes the loss between the actual and predicted values.
     * 
     * @param actual    The actual values.
     * @param predicted The predicted values.
     * @return The computed loss.
     */
    double compute(double[] actual, double[] predicted);

    /**
     * Computes the derivative of the loss with respect to the predicted values.
     * 
     * @param actual    The actual values.
     * @param predicted The predicted values.
     * @return The derivatives.
     */
    double[] derivative(double[] actual, double[] predicted);
}
