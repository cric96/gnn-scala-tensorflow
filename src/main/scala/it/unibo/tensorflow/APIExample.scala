package it.unibo.tensorflow

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.layers.{Linear, Loss}
import org.platanios.tensorflow.api.learn.{Mode, TRAINING}
import org.platanios.tensorflow.api.tensors.Tensor

object APIExample extends App {
  val s = Session()
  implicit class RichOutput[T](out : Output[T]) {
    def get : Tensor[T] = s.run(fetches = out)
  }
  // Tensor is compiled checked
  val zeros : Tensor[Float] = Tensor.zeros[Float](Shape(2, 2, 9, 6)) // equivalent to tf.zeros((2, 2, 9, 6), dtype=tf.float32)
  // val zeroError : Tensor[Int] = Tensor.zeros[Float](Shape(2, 5)) error!!
  println(zeros.summarize())
  val tensor = Tensor(1, 2, 3) // equivalent to tf.tensor(1, 2, 3)
  val matrix = Tensor[Float](Tensor(1, 2, 3), Tensor(1, 2, 3), Tensor(1, 2, 3)) // equivalent to tf.tensor([1,2,3], [1, 2, 3], [1, 2, 3], dtype=tf.float32)
  // Operations:
  val sum = tensor + tensor
  val mul = tensor * tensor
  // It has a similar tensorflow api through tf
  println(tf.matmul(matrix, matrix).get.summarize())
  // Slicing
  val slice = zeros(::, ::, 0 :: 2, ::)
  println(slice.summarize())
  // Neural Network Layer
  val dense = Linear[Float]("Layer_0", 32) // Layer type must be explicit
  val input = Tensor.zeros[Float](Shape(1, 32))
  // It is a strange Scala mechanism, called implicit.. If there is a contextual value, you can avoid to pass it through the method call
  // For example:
  val result = {
    implicit val mode : Mode = TRAINING
    dense(input)
  }
  // Or you can pass it explicitly
  dense(input)(TRAINING)
  // It seems strange, but in several context it is very useful..
  println("Layer result shape = " + result.shape)
  val loss = Loss.API.L2Loss[Float, Float]("mse")
  val lossResult = loss((input, result))(TRAINING)
  val optimizer = tf.train.GradientDescent(1e-6f)
  val gradientsAndVariables = tf.train.GradientDescent(1e-6f)
    .computeGradients(result, colocateGradientsWithOps = false)
  // Idea is confused by the compiler find the right implicit..
  val appliedGradient = optimizer.applyGradients(gradientsAndVariables)
}
