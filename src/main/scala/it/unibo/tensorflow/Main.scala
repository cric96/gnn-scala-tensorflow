package it.unibo.tensorflow

import org.platanios.tensorflow.api.learn.layers.{Loss, ReLU}
import org.platanios.tensorflow.api.learn.{INFERENCE, Mode, TRAINING}
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}
import org.platanios.tensorflow.api.{Output, Tensor, _}
// N.B. intellidea compiler seems to have problems with implicit resolution. Even if there are errors, the program is correct..
object Main extends App {
  // Some constant
  private val inputSize = 1
  private val hiddenFeatureSize = 3
  private val outputSize = 1
  private val denseLayer0 = Dense("0", hiddenFeatureSize, inputSize)
  private val denseLayer1 = Dense("1", hiddenFeatureSize, hiddenFeatureSize)
  private val outputLayer = Dense("output", outputSize, hiddenFeatureSize + inputSize)
  private val activation = ReLU[Float]("Activation0")
  // Utility to compose with dense function
  private val activationFn: Output[Float] => Output[Float] = output => activation(output)(TRAINING)
  // It is like denseLayer => ReLU => denseLayer1 => ReLU
  private val mlp = activationFn compose denseLayer1 compose activationFn compose denseLayer0
  // Train utils
  private val optimizer = GradientDescent(0.009f)
  private val epoch = 300
  implicit val mode: Mode = INFERENCE // Contextual abstraction, here the
  // Current tensorflow session
  private implicit val session: Session = Session()
  // Initialize variables
  tf.globalVariablesInitializer().eval()
  // Utility to perform a forward gnn like pass
  def forward(input: Output[Float], adjacencyMatrix: Output[Float])(implicit
      mode: Mode
  ): Output[Float] = {
    val hidden = GNN.layer(input, adjacencyMatrix, mlp, activation) // learn representation
    val concatenation =
      tf.concatenate(Seq(hidden, input), axis = 1) // concat representation with feature vector
    outputLayer(concatenation) // compute the right value
  }
  // Train cycle
  def train(optimizer: Optimizer, epochs: Int): Unit = {
    val loss = Loss.API.L2Loss[Float, Float]("loss")
    implicit val mode: Mode = TRAINING
    (0 to epochs).foreach(i => {
      val output = forward(Dataset.nodes, Dataset.edges) // forward pass
      val lossResult = loss((output, Dataset.groundTruth)) // compute loss
      println(s"Epoch $i,")
      val gradients = optimizer.minimize(lossResult)
      gradients.eval()
      println("loss" + lossResult.value().summarize())
    })
  }
  // API Call example
  train(optimizer, epoch)
  // "Validation"
  val result = forward(Dataset.nodes, Dataset.edges)(INFERENCE)
  println(result.value().summarize())
}
