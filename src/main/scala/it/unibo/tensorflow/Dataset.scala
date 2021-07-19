package it.unibo.tensorflow

import org.platanios.tensorflow.api.Tensor

object Dataset {

  /** Graph example:
    * (1, 0) - (0, 1)
    *  |        |
    * (0, 2) - (0, 3)
    */
  val nodes = Tensor(
    Tensor(1f), // node 0
    Tensor(0f), // node 1
    Tensor(0f), // node 2
    Tensor(0f) // node 3
  )
  val edges = Tensor( // With self loops
    Tensor(1f, 1f, 1f, 0f), // node 0
    Tensor(1f, 1f, 0f, 1f), // node 1
    Tensor(1f, 0f, 1f, 1f), // node 2
    Tensor(0f, 1f, 1f, 1f) // node 3
  )
  // hop count towards node 0
  val groundTruth = Tensor(
    Tensor(0f),
    Tensor(1f),
    Tensor(1f),
    Tensor(2f)
  )
}
