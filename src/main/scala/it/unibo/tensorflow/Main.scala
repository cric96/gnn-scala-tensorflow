package it.unibo.tensorflow

import it.unibo.tensorflow.GNN.Dense
import org.platanios.tensorflow.api.learn.TRAINING
import org.platanios.tensorflow.api.learn.layers.{Identity, Linear, Loss, ReLU}
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}
import org.platanios.tensorflow.api.tf.learn.Layer
import org.platanios.tensorflow.api.{Output, Tensor, _}

object Main extends App {
  val inputSize = 1
  val hiddenFeatureSize = 32
  val outputSize = 1
  val weight0 = tf.variable[Float]("W0", Shape(1, hiddenFeatureSize), tf.RandomUniformInitializer())
  val weight1 = tf.variable[Float]("W1", Shape(hiddenFeatureSize, outputSize), tf.RandomUniformInitializer())
  val denseLayer0 = Dense(weight0)
  val denseLayer1 = Dense(weight1)
  val activation = ReLU[Float]("Activation0")
  val identity : Layer[Output[Float], Output[Float]] = Identity[Output[Float]]("Output")
  def train(optimizer : Optimizer  , epochs : Int) : Unit = {
    implicit val mode = TRAINING
    (0 to epochs).foreach(i => {
      val hidden = GNN.layer(Dataset.nodes, Dataset.edges, denseLayer0, activation)
      val output = GNN.layer(hidden, Dataset.edges, denseLayer1, identity)
      val loss = Loss.API.L2Loss[Float, Float]("loss")
      val lossResult = loss((Dataset.nodes, output))
      println(s"Epoch $i,")
      val gradients = optimizer.minimize(lossResult)
      session.run(targets = gradients)
      val lossResultTensor = session.run(fetches = lossResult)
      println(session.run(fetches = output).summarize())
      println(lossResultTensor.summarize())
    })
  }

  val session = Session()
  val weights =
  session.run(targets = tf.globalVariablesInitializer())
  val optimizer = GradientDescent(0.003f)
  val epoch = 100
  train(optimizer, 100)
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
    Tensor(1f, 1f, 0f, 1f), // node 3
  )
  // hop count towards node 0
  val groundTruth = Tensor(
    Tensor(0f),
    Tensor(1f),
    Tensor(1f),
    Tensor(2f),
  )
}
