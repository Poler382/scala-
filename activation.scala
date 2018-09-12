import math._


class Sigmoid() extends Layer {
  var ys = List[Array[T]]()

  def push(y: Array[T]) = {
    ys ::= y;
    y
  }

  def pop() = {
    val y = ys.head;
    ys = ys.tail;
    y
  }

  def sigmoid(x: Double) = 1 / (1 + math.exp(-x))

  def forward(x: Array[T]) = {
    push(x.map(sigmoid))
  }

  def backward(d: Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => d(i) * y(i) * (1d - y(i))).toArray
  }

  def update(){ reset()}

  def reset(){ ys = List[Array[T]]()}
 
  override def save(fn: String) {}

  override def load(fn: String) {}
}

class Tanh() extends Layer {
  var ys = List[Array[T]]()

  def push(y: Array[T]) = {
    ys ::= y;
    y
  }

  def pop() = {
    val y = ys.head;
    ys = ys.tail;
    y
  }

  def forward(x: Array[T]) = {
    push(x.map(math.tanh))
  }

  def backward(d: Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => d(i) * (1d - y(i) * y(i))).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}


class ReLU() extends Layer {
  var ys = List[Array[T]]()

  def push(y: Array[T]) = {
    ys ::= y;
    y
  }

  def pop() = {
    val y = ys.head;
    ys = ys.tail;
    y
  }

  def forward(x: Array[T]) = {
    push(x.map(a => math.max(a, 0)))
  }

  def backward(d: Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => if (y(i) > 0) d(i) else 0d).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }

}

class LeakyReLU(val alpha: Double) extends Layer {
  var ys = List[Array[T]]()

  def push(y: Array[T]) = {
    ys ::= y;
    y
  }

  def pop() = {
    val y = ys.head;
    ys = ys.tail;
    y
  }

  def forward(x: Array[T]) = {
    push(x.map(a => if (a > 0) a else alpha * a))
  }

  def backward(d: Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => if (y(i) > 0) d(i) else alpha * d(i)).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}
