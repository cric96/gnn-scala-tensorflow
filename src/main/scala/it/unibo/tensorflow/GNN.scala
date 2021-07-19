package it.unibo.tensorflow

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.tf.learn.Layer

object GNN {
  type GraphShape = Output[Float]
  // H_i = sigma(A*H_(i-1)*W
  def layer(
      features: GraphShape,
      adjacentMatrix: GraphShape,
      transform: GraphShape => GraphShape,
      activation: Layer[GraphShape, GraphShape]
  )(implicit mode: Mode): GraphShape = {
    val localNodeApplication = transform(features) //  H_(i -1) * W
    val neighbourAggregation = tf.matmul(adjacentMatrix, localNodeApplication) // A * H_(i-1) * W
    activation(neighbourAggregation) // sigma(A*H_(i-1)*W)
  }
}
