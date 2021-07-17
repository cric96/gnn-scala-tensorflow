package it.unibo.tensorflow

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.tf.learn.Layer

object GNN {
  type GraphShape = Output[Float]
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

  case class Dense(variable: Variable[Float]) {
    def apply(output: Output[Float]): Output[Float] = tf.matmul(output, variable)
  }
}
