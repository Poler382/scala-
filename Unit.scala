import math._
import breeze.linalg._
import kata._

/*
l8-l132:Affine
l134-262:Convolition3D
 */

class Affine(val xn: Int, val yn: Int,
  val eps:T= 0.001, val rho1:T = 0.9, val rho2:T= 0.999) extends Layer {
  val rand = new scala.util.Random(0)

  var W = Array.ofDim[T](xn * yn).map(_ => rand.nextGaussian * 0.01:T)
  var b = Array.ofDim[T](yn)
  var dW = Array.ofDim[T](xn * yn)
  var db = Array.ofDim[T](yn)
  var n = 0

  def windex(i: Int, j: Int) = i * xn + j

  var xs = List[Array[T]]()

  def push(x: Array[T]) = {
    xs ::= x;
    x
  }

  def pop() = {
    val x = xs.head;
    xs = xs.tail;
    x
  }

  def forward(x: Array[T]) = {
    push(x)
    val y = Array.ofDim[T](yn)
    for (i <- 0 until yn) {
      for (j <- 0 until xn) {
        y(i) += W(windex(i, j)) * x(j)
      }
      y(i) += b(i)
    }
    y
  }

  def backward(d: Array[T]) = {
    val x = pop()
    n += 1

    for (i <- 0 until yn; j <- 0 until xn) {
      dW(windex(i, j)) += d(i) * x(j)
    }

    for (i <- 0 until yn) {
      db(i) += d(i)
    }

    val dx = Array.ofDim[T](xn)
    for (j <- 0 until yn; i <- 0 until xn) {
      dx(i) += W(windex(j, i)) * d(j)
    }

    dx
  }

  def update() {
    for (i <- 0 until dW.size) {
      dW(i) /= n
    }
    for (i <- 0 until db.size) {
      db(i) /= n
    }
    update_adam()
    reset()
  }


  var adam_W = new Adam(W.size, eps, rho1, rho2)
  var adam_b = new Adam(b.size, eps, rho1, rho2)

  def update_adam() {
    adam_W.update(W, dW)
    adam_b.update(b, db)
  }

  var lr = 0.001:T

  def update_sgd() {
    for (i <- 0 until W.size) {
      W(i) -= lr * dW(i)
    }

    for (i <- 0 until b.size) {
      b(i) -= lr * db(i)
    }
  }

  def reset() {
    for (i <- 0 until dW.size) {
      dW(i) = 0d
    }
    for (i <- 0 until db.size) {
      db(i) = 0d
    }
    xs = List[Array[T]]()
    n = 0
  }

  override def save(fn: String) {
    val pw = new java.io.PrintWriter(fn)
    for (i <- 0 until W.size) {
      pw.write(W(i).toString)
      if (i != W.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    for (i <- 0 until b.size) {
      pw.write(b(i).toString)
      if (i != b.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }

  override def load(fn: String) {
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    W = f(0).split(",").map(_.toDouble).toArray
    b = f(1).split(",").map(_.toDouble).toArray
  }
}

class Convolution3D(
  val KW:Int,val IH:Int,val IW:Int,val IC:Int,val OC:Int,
  val ss:Int=1,
  val e:T = 0.01:T,
  val p1:T = 0.9:T)extends Layer {
  val OH = 1 + (IH-KW)/ss //IH - KW + 1
  val OW = 1 + (IW-KW)/ss // w - kw + 1
  val rand = new scala.util.Random(0)
  var t=0
  var p1_t=1d
  var p2_t=1d
  var K = Array.ofDim[T](OC,IC,KW*KW).map(_.map(_.map(a => rand.nextGaussian*0.01)))
  var V_D = List[Array[T]]()
  var D_K = Array.ofDim[T](OC,IC,KW*KW)
  var s = Array.ofDim[T](OC,IC,KW*KW)
  var r = Array.ofDim[T](OC,IC,KW*KW)
  var n = 0d

  def iindex(i:Int, j:Int, k:Int) = i * IH*IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH*OW + j * OW + k
 
  def push(V:Array[T]) = { V_D ::= V; V }
  def pop() = { val V = V_D.head; V_D = V_D.tail; V }
 
  def forward(V:Array[T]) = {
    push(V)
    val Z = Array.ofDim[T](OC*OH*OW)//1 + (IW-KW)/ss)

    for(i<-0 until OC ; j<-0 until OH ; k<-0 until OW){
      var s = 0d
      for(l<-0 until IC ; m<-0 until KW ; n<-0 until KW){
        s +=  V(iindex(l,j*ss+m,k*ss+n)) * K(i)(l)(m*KW+n)
      }
      Z(oindex(i,j,k)) = s
    }
    Z
  }

  def backward(G:Array[T]) = {

    val x = pop()
    n += 1d

    for(i<-0 until OC ;  j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      var s = 0d
      for(m<-0 until  OH ; n<-0 until OW){
        s += G(oindex(i,m,n)) * x(iindex(j,m*ss+k,n*ss+l))
      }

      D_K(i)(j)(k*KW+l) = s
    }

    val dV = Array.ofDim[Double](IC * IH * IW)

    for(i<-0 until IC ; j<-0 until IH ; k<-0 until IW){
      var s1=0d
      for(l <- math.max((j-KW)/ss,0) until math.min(j/ss+1,OH) ; m <- 0 until KW){
        if(j==l*ss+m){
          for(n <- math.max((k-KW)/ss,0) until math.min(k/ss+1,OW) ; p<-0 until KW){
            if(k==n*ss+p)
              for(q<-0 until OC)
                s1 += K(q)(i)(m*KW+p)*G(oindex(q,l,n))
          }
        }
      }
      dV(iindex(i,j,k)) = s1
    }
    dV
  }

  def update() {

    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW*KW){
      D_K(i)(j)(k) = D_K(i)(j)(k)/n
    }
    val p2 = 0.999
    val delta = 0.00000001

    var s_h = Array.ofDim[T](OC,IC,KW*KW)
    var r_h = Array.ofDim[T](OC,IC,KW*KW)
    
    t += 1

    p1_t = p1 * p1_t
    p2_t = p2 * p2_t

    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW*KW){
      var D_s = 0d//ΔΘ

      s(i)(j)(k) = p1*s(i)(j)(k)+(1-p1)*D_K(i)(j)(k)
      r(i)(j)(k) = p2*r(i)(j)(k)+(1-p2)*D_K(i)(j)(k)*D_K(i)(j)(k)

      s_h(i)(j)(k) = s(i)(j)(k)/(1-p1_t)
      r_h(i)(j)(k) = r(i)(j)(k)/(1-p2_t)

      D_s = -1*e*s_h(i)(j)(k)/(math.sqrt(r_h(i)(j)(k))+delta)
      K(i)(j)(k) = K(i)(j)(k) + D_s
     
    }
    n=0d
    reset()
  }

  def update2(){
    val lr = 0.9
    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW*KW)
      K(i)(j)(k) -= lr * D_K(i)(j)(k)
    reset()
  }

  def reset() {
    D_K = Array.ofDim[T](OC,IC,KW*KW)
  }

  override def save(fn:String) {
    val pw = new java.io.PrintWriter(fn)
    for(i <- 0 until OC ; j <- 0 until IC ; k <- 0 until KW*KW) {
      pw.write(K(i)(j)(k).toString)
      if(i != OC - 1 || j != IC-1 || k != KW*KW-1 ) {
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }

  override def load(fn:String) {
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    val tmp = f(0).split(",").map(_.toDouble).toArray
    for(i <- 0 until OC ; j <- 0 until IC ; k <- 0 until KW*KW) {
      K(i)(j)(k) = tmp(i*IC*KW*KW+j*KW*KW+k)
    }
  }
}


class Upsampling(val IC: Int, val IH: Int, val IW: Int, val BH: Int, val BW: Int) extends Layer {
  val OH = IH * BH
  val OW = IW * BW
  val OC = IC

  def iindex(i: Int, j: Int, k: Int) = i * IH * IW + j * IW + k

  def oindex(i: Int, j: Int, k: Int) = i * OH * OW + j * OW + k

  def forward(X: Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for (i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      Z(oindex(i, j, k)) = X(iindex(i, j / BH, k / BW))
    }
    Z
  }

  def backward(d: Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for (i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      D(iindex(i, j / BH, k / BW)) += d(oindex(i, j, k))
    }
    D
  }

  def update() {}

  def reset() {}
}

class StridedPadding(val IC: Int, val IH: Int, val IW: Int, val S: Int) extends Layer {
  val OH = S * IH + S - 1
  val OW = S * IW + S - 1
  val OC = IC

  def iindex(i: Int, j: Int, k: Int) = i * IH * IW + j * IW + k

  def oindex(i: Int, j: Int, k: Int) = i * OH * OW + j * OW + k

  def forward(X: Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for (i <- 0 until IC; j <- 0 until IH; k <- 0 until IW) {
      Z(oindex(i, S - 1 + j * S, S - 1 + k * S)) = X(iindex(i, j, k))
    }
    Z
  }

  def backward(d: Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for (i <- 0 until IC; j <- 0 until IH; k <- 0 until IW) {
      D(iindex(i, j, k)) = d(oindex(i, S - 1 + j * S, S - 1 + k * S))
    }
    D
  }

  def update() {}

  def reset() {}
}

class Subsampling(val IC: Int, val IH: Int, val IW: Int, val BH: Int, val BW: Int) extends Layer {
  val OH = IH / BH
  val OW = IW / BW
  val OC = IC

  def iindex(i: Int, j: Int, k: Int) = i * IH * IW + j * IW + k

  def oindex(i: Int, j: Int, k: Int) = i * OH * OW + j * OW + k

  def forward(X: Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for (i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      Z(oindex(i, j, k)) = X(iindex(i, j * BH, k * BW))
    }
    Z
  }

  def backward(d: Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for (i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      D(iindex(i, j * BH, k * BW)) += d(oindex(i, j, k))
    }
    D
  }

  def update() {}

  def reset() {}
}

class Pooling(val BW: Int, val IC: Int, val IH: Int, val IW: Int) extends Layer {
  val OH = IH / BW
  val OW = IW / BW
  val OC = IC
  var masks = List[Array[T]]()

  def push(x: Array[T]) = {
    masks ::= x;
    x
  }

  def pop() = {
    val mask = masks.head;
    masks = masks.tail;
    mask
  }

  def iindex(i: Int, j: Int, k: Int) = i * IH * IW + j * IW + k

  def oindex(i: Int, j: Int, k: Int) = i * OH * OW + j * OW + k

  def forward(X: Array[T]) = {
    val mask = push(Array.ofDim[T](IC * IH * IW))
    val Z = Array.ofDim[T](OC * OH * OW)
    for (i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      var v = Double.NegativeInfinity
      var row_max = -1
      var col_max = -1
      for (m <- 0 until BW; n <- 0 until BW if v < X(iindex(i, j * BW + m, k * BW + n))) {
        row_max = j * BW + m
        col_max = k * BW + n
        v = X(iindex(i, j * BW + m, k * BW + n))
      }
      mask(iindex(i, row_max, col_max)) = 1
      Z(oindex(i, j, k)) = v
    }
    Z
  }

  def backward(d: Array[T]) = {
    val mask = pop()
    val D = Array.ofDim[T](mask.size)
    for (i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      for (m <- 0 until BW; n <- 0 until BW if mask(iindex(i, j * BW + m, k * BW + n)) == 1) {
        D(iindex(i, j * BW + m, k * BW + n)) = d(oindex(i, j, k))
      }
    }
    D
  }

  def update() {}

  def reset() {
    masks = List[Array[T]]()
  }
}
class Dropout(var dr: Double) extends Layer {
  var masks = List[Array[T]]()

  def push(mask: Array[T]) = {
    masks ::= mask;
    mask
  }

  def pop() = {
    val mask = masks.head;
    masks = masks.tail;
    mask
  }

  val rand = new util.Random(0)

  def forward(x: Array[T]) = {
    if (is_test) {
      x.map(_ * (1 - dr))
    } else {
      val mask = push(Array.ofDim[T](x.size))
      for (i <- 0 until x.size) {
        if (rand.nextDouble > dr) {
          mask(i) = 1d
        }
      }
      x.zip(mask).map { case (a, b) => a * b }.toArray
    }
  }

  def backward(d: Array[T]) = {
    val mask = pop()
    (0 until d.size).map(i => if (mask(i) > 0) d(i) else 0d).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    masks = List[Array[T]]()
  }
}

class Ident() extends Layer {
  def forward(x: Array[T]) = x

  def backward(d: Array[T]) = d

  def update() {}

  def reset() {}
}

class ZeroPadding(val IC: Int, val IH: Int, val IW: Int, P: Int) extends Layer {
  val OH = IH + 2 * P
  val OW = IW + 2 * P
  val OC = IC

  def iindex(i: Int, j: Int, k: Int) = i * IH * IW + j * IW + k

  def oindex(i: Int, j: Int, k: Int) = i * OH * OW + j * OW + k

  def forward(x: Array[T]) = {
    val y = new Array[T](OC * OH * OW)
    for (c <- 0 until IC; i <- 0 until IH; j <- 0 until IW) {
      y(oindex(c, i + P, j + P)) = x(iindex(c, i, j))
    }
    y
  }

  def backward(d: Array[T]) = {
    val d1 = new Array[T](IC * IH * IW)
    for (c <- 0 until IC; i <- 0 until IH; j <- 0 until IW) {
      d1(iindex(c, i, j)) = d(oindex(c, i + P, j + P))
    }
    d1
  }

  def update() {}

  def reset() {}
}



class Adam(val n: Int,
  val eps:T= 0.0002:T,
  val rho1:T= 0.5:T,
  val rho2:T= 0.999:T) extends layerType {

  val delta = 1e-8
  var rho1t = 1:T
  var rho2t = 1:T
  var s = Array.ofDim[T](n)
  var r = Array.ofDim[T](n)

  def update(K: Array[T], dK: Array[T]) = {
    var nK = Array.ofDim[T](K.size)
    rho1t *= rho1
    rho2t *= rho2
    val rho1tr = 1 / (1 - rho1t)
    val rho2tr = 1 / (1 - rho2t)
    for (i <- 0 until K.size) {
      s(i) = rho1 * s(i) + (1 - rho1) * dK(i)
      r(i) = rho2 * r(i) + (1 - rho2) * dK(i) * dK(i)
      val d = (s(i) * rho1tr) / (math.sqrt(r(i) * rho2tr) + delta)
      K(i) = K(i) - eps * d
    }
  }
}





class Softplus() extends Layer {
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

  def softplus(x:T) = x + math.log(1d + math.exp(-x)):T

  def sigmoid(x:T) = 1 / (1d + math.exp(-x)):T

  def forward(x: Array[T]) = {
    push(x)
    x.map(softplus)
  }

  def backward(d: Array[T]) = {
    val x = pop()
    (0 until d.size).map(i => d(i) * sigmoid(x(i))).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }

}



class lstm(
            val In: Int,
            val Out: Int,
            val hsize: Int
          ) extends Layer {

  var hiddenGate = List(new Affine(In + hsize, Out), new Tanh())
  var inputGate  = List(new Affine(In + hsize, Out), new Sigmoid())
  var forgetGate = List(new Affine(In + hsize, Out), new Sigmoid())
  var outputGate = List(new Affine(In + hsize, Out), new Sigmoid())
  var cList = new Stack[Array[T]]()
  var hList = new Stack[Array[T]]()
  var iList = new Stack[Array[T]]()
  var dList = new Stack[Array[T]]()
  var oList = new Stack[Array[T]]()
  var fList = new Stack[Array[T]]()
  var tanhList = new Stack[Array[T]]()
  var itList = new Stack[Array[T]]()
  var h_hatList = new Stack[Array[T]]()
  val L = new ML()

  val t = new Tanh()
  hList.push(new Array[T](Out))
  cList.push(new Array[T](Out))

  def forward(xs: Array[T]) = {
    var h_hat = L.forwards(hiddenGate, xs ++ hList.head)
    var it = L.forwards(inputGate, xs ++ hList.head)
    var ft = L.forwards(forgetGate, xs ++ hList.head)
    var ot = L.forwards(outputGate, xs ++ hList.head)

    val c1 = it.zip(h_hat).map { case (a, b) => a * b }
    val c2 = ft.zip(cList.head).map { case (a, b) => a * b }
    val c = c1.zip(c2).map { case (a, b) => a + b }

    cList.push(c)
    tanhList.push(t.forward(c))
    itList.push(it)
    h_hatList.push(h_hat)
    oList.push(ot)
    fList.push(ft)

    val h_t = ot.zip(tanhList.head).map { case (a, b) => a * b }

    hList.push(h_t)
    h_t
  }

  var rList = new Stack[Array[T]]()
  rList.push(new Array[T](Out))

  def backward(d: Array[T]) = {
    val ds = d.zip(rList.head).map { case (a, b) => a + b }
    val b_ot = L.backwards(outputGate, ds.zip(tanhList.pop()).map { case (a, b) => a * b })
    val b_tanh = t.backward(ds.zip(oList.pop()).map { case (a, b) => a * b })
    val bc = cList.pop().zip(b_tanh).map { case (a, b) => a + b }
    val bf = L.backwards(forgetGate, cList.head.zip(bc).map { case (a, b) => a * b })
    val m = cList.push(fList.head.zip(bc).map { case (a, b) => a * b })
    val bh_hat = L.backwards(hiddenGate, bc.zip(itList.pop).map { case (a, b) => a * b })
    val bi = L.backwards(inputGate, bc.zip(h_hatList.head).map { case (a, b) => a * b })

    var dxh = List[T]()

    for (i <- 0 until bi.size) {
      val temp = bi(i) + bh_hat(i) + b_ot(i) + bf(i)
      dxh ::= temp
    }

    dxh = dxh.reverse

    val preh = dxh.drop(hsize)

    rList.push(preh.toArray)

    d
  }

  def update() {
    L.updates(hiddenGate)
    L.updates(inputGate)
    L.updates(forgetGate)
    L.updates(outputGate)
  }

  def reset() {
    L.resets(hiddenGate)
    L.resets(inputGate)
    L.resets(forgetGate)
    L.resets(outputGate)
  }

  override def save(fn: String): Unit = {
    L.saves(hiddenGate, fn + "_hiddenGate")
    L.saves(inputGate, fn + "_inputGate")
    L.saves(forgetGate, fn + "_forgetGate")
    L.saves(outputGate, fn + "_outputGate")
  }

  override def load(fn: String): Unit = {
    L.load(hiddenGate, fn + "_hiddenGate")
    L.load(inputGate, fn + "_inputGate")
    L.load(forgetGate, fn + "_forgetGate")
    L.load(outputGate, fn + "_outputGate")
  }
}




class softMax() extends Layer {
  val Es = new Stack[Array[Double]]()

  def forward(xs: Array[T]) = {

    val ex = xs.map(math.exp(_))
    Es.push(ex)
    val sum = 1d / ex.sum
    val y = ex.map(_ * sum)
    y
  }

  def backward(ds: Array[T]) = {
    val ex = Es.pop()
    val exsum = ex.sum
    val de_sum = ds.zip(ex).map { case (a, b) => a * b }.sum

    var x = Array.ofDim[T](ds.size)

    for (i <- 0 until ds.size) {
      x(i) = ((ds(i) / exsum) - de_sum / (exsum * exsum)) * ex(i)
    }
    x
  }

  def update() {
    reset()
  }

  def reset() {
    Es.reset()
  }

  override def save(fn: String) {}

  override def load(fn: String) {}
}
