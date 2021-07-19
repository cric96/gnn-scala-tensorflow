package it.unibo

import org.platanios.tensorflow.api.ops.UntypedOp
import org.platanios.tensorflow.api.{Output, Session, Tensor}

package object tensorflow {
  // TYPE ENRICHMENT, a.k.a pimp my library
  implicit class RichOutput[T](out: Output[T])(implicit session: Session) {
    def value(): Tensor[T] = session.run(fetches = out)
  }
  implicit class RichOp[T](out: UntypedOp)(implicit session: Session) {
    def eval(): Unit = session.run(targets = out)
  }
}
