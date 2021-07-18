package it.unibo.tensorflow

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.variables.{OnesInitializer, RandomNormalInitializer, RandomUniformInitializer, ZerosInitializer}
import org.platanios.tensorflow.api.tf.learn.Layer

object GNN {
  type GraphShape = Output[Float]
  // H_i = sigma(A*H_(i-1)*W=
  def layer(
    features: GraphShape,
    adjacentMatrix: GraphShape,
    transform: Dense,
    activation: Layer[GraphShape, GraphShape]
  )(implicit mode : Mode): GraphShape = {
    val localNodeApplication = transform(features)
    val neighbourAggregation = tf.matmul(adjacentMatrix, localNodeApplication)
    activation(neighbourAggregation)
  }

  case class Dense(name : String, units : Int, inputSize : Int) {
    private val weights : Variable[Float] = tf.variable[Float](s"W-$name", Shape(inputSize, units), RandomNormalInitializer())
    private val bias : Variable[Float] = tf.variable[Float](s"B-$name", Shape(units), ZerosInitializer)
    def apply(input: Output[Float]): Output[Float] = tf.matmul(input, weights) + bias
  }
}
