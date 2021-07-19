package it.unibo.tensorflow

import org.platanios.tensorflow.api.ops.variables.{RandomNormalInitializer, ZerosInitializer}
import org.platanios.tensorflow.api.{Output, Shape, Variable, tf}

case class Dense(name: String, units: Int, inputSize: Int)
    extends ((Output[Float]) => Output[Float]) {
  private val weights: Variable[Float] =
    tf.variable[Float](s"W-$name", Shape(inputSize, units), RandomNormalInitializer())
  private val bias: Variable[Float] = tf.variable[Float](s"B-$name", Shape(units), ZerosInitializer)
  def apply(input: Output[Float]): Output[Float] = tf.matmul(input, weights) + bias
}
