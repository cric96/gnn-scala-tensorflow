package it.unibo

import org.platanios.tensorflow.api.ops.UntypedOp
import org.platanios.tensorflow.api.{Output, Session, Tensor}

package object tensorflow {
  implicit class RichOutput[T](out : Output[T])(implicit session : Session) {
    def value() : Tensor[T] = session.run(fetches = out)
    def eval() : Unit = value()
  }
  implicit class RichOp[T](out : UntypedOp)(implicit session : Session) {
    def eval() : Unit =  session.run(targets = out)
  }
}
