package it.unibo.tensorflow

import it.unibo.tensorflow.GNN.Dense
import org.platanios.tensorflow.api.learn.{INFERENCE, Mode, TRAINING}
import org.platanios.tensorflow.api.learn.layers.{Identity, Linear, Loss, ReLU}
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}
import org.platanios.tensorflow.api.tf.learn.Layer
import org.platanios.tensorflow.api.{Output, Tensor, _}

object Main extends App {
  val inputSize = 1
  val hiddenFeatureSize = 3
  val outputSize = 1
  val denseLayer0 = Dense("0", hiddenFeatureSize, inputSize)
  val denseLayer1 = Dense("1", hiddenFeatureSize, hiddenFeatureSize)
  val outputLayer = Dense("output", outputSize, hiddenFeatureSize + inputSize)
  val activation = ReLU[Float]("Activation0")
  val activationFn : Output[Float] => Output[Float] = output => activation(output)(TRAINING)
  val mlp = denseLayer1 compose activationFn compose denseLayer0
  implicit val session : Session = Session()

  def forward()(implicit mode : Mode) : Output[Float] = {
    val hidden = GNN.layer(Dataset.nodes, Dataset.edges, mlp, activation) // learn representation
    val concatenation = tf.concatenate(Seq(hidden, (Dataset.nodes : Output[Float])), 1) // concat representation with feature vector
    outputLayer(concatenation) // compute the right value
  }
  def train(optimizer : Optimizer, epochs : Int) : Unit = {
    val loss = Loss.API.L2Loss[Float, Float]("loss")
    implicit val mode = TRAINING
    (0 to epochs).foreach(i => {
      val output = forward()
      val lossResult = loss((output, Dataset.groundTruth)) // compute loss
      println(s"Epoch $i,")
      val gradients = optimizer.minimize(lossResult)
      gradients.eval()
      println("loss" + lossResult.value().summarize())
    })
  }
  val optimizer = GradientDescent(0.009f)
  tf.globalVariablesInitializer().eval()
  val epoch = 300
  train(optimizer, epoch)
  val result = forward()(INFERENCE)
  println(result.value().summarize())
}

object Dataset {
  /**
   * Graph example:
   * (1, 0) - (0, 1)
   *  |        |
   * (0, 2) - (0, 3)
   */
  val nodes = Tensor(
    Tensor(1f), // node 0
    Tensor(0f), // node 1
    Tensor(0f), // node 2
    Tensor(0f), // node 3
  )
  val edges = Tensor( // With self loops
    Tensor(1f, 1f, 1f, 0f), // node 0
    Tensor(1f, 1f, 0f, 1f), // node 1
    Tensor(1f, 0f, 1f, 1f), // node 2
    Tensor(0f, 1f, 1f, 1f), // node 3
  )
  // hop count towards node 0
  val groundTruth = Tensor(
    Tensor(0f),
    Tensor(1f),
    Tensor(1f),
    Tensor(2f),
  )
}
