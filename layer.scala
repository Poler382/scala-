import kata._
abstract class Layer {
  type T = Float
  var is_test = false
  def forward(x: Array[T]): Array[T]
  def backward(x: Array[T]): Array[T]
  def forward(xs: Array[Array[T]]): Array[Array[T]] = {
    xs.map(forward)
  }
  def backward(ds: Array[Array[T]]): Array[Array[T]] = {
    ds.reverse.map(backward).reverse
  }
  def update(): Unit
  def reset(): Unit
  def save(fn: String) {}
  def load(fn: String) {}
}
/*
abstract class layerType{
  type T = Float
}*/

object kata{
  type T = Float
  //すべてのファイルにimportする
}


class Image(){
  def rgb(im: java.awt.image.BufferedImage, i: Int, j: Int) = {
    val c = im.getRGB(i, j)
    Array(c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff)
  }

  def pixel(r: Int, g: Int, b: Int) = {
    val a = 0xff
    ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
  }

  def read(fn: String) = {
    val im = javax.imageio.ImageIO.read(new java.io.File(fn))
    (for (i <- 0 until im.getHeight; j <- 0 until im.getWidth)
      yield rgb(im, j, i)).toArray.grouped(im.getWidth).toArray
  }

  def write(fn: String, b: Array[Array[Array[Int]]]) = {
    val w = b(0).size
    val h = b.size
    val im = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
    for (i <- 0 until im.getHeight; j <- 0 until im.getWidth) {
      im.setRGB(j, i, pixel(b(i)(j)(0), b(i)(j)(1), b(i)(j)(2)));
    }
    javax.imageio.ImageIO.write(im, "png", new java.io.File(fn))
  }

  def make_image(ys: Array[Array[T]], NW: Int, NH: Int, H: Int, W: Int) = {
    val im = Array.ofDim[Int](NH * H, NW * W, 3)
    val ymax = ys.flatten.max
    val ymin = ys.flatten.min

    def f(a: T) = ((a - ymin) / (ymax - ymin) * 255).toInt

    for (i <- 0 until NH; j <- 0 until NW) {
      for (p <- 0 until H; q <- 0 until W; k <- 0 until 3) {
        //k * H * W
        im(i * H + p)(j * W + q)(k) = f(ys(i * NW + j)(p * W + q))
      }
    }
    im
  }
  def RGBto3DArray(image:Array[T],h:Int,w:Int)={
    /*
     rgbrgb...と並ぶ配列を
     Array(Array(rrr),Array(ggg),Array(bbb))並び換える関数
     主にチャンネルをわけるのに使用する
     */
    var R = List[T]()
    var G = List[T]()
    var B = List[T]()

    for(im <- 0 until image.size){
      if(im % 3 == 0){
        R ::= image(im)
      }
      else if(im % 3 == 1){
        G ::= image(im)
      }
      else if(im % 3 == 2){
        B ::= image(im)
      }
    }

    Array(R.reverse.toArray,G.reverse.toArray,B.reverse.toArray)

  }
  def RGBtoGray (image:Array[T],h:Int,w:Int)={
    var gray = List[T]()
    var sum= 0:T
    /*
     rgbrgb...と並ぶ配列を各画素毎に合計をとりグレースケールを取る
     Array(gray,gray,gray)の一次元配列で返す
     主にチャンネルをわけずデータの圧縮に使用する
     */
    for(i <- 0 until image.size){
      if(i%3 == 0 && i != 0){
        gray ::= sum/3
      }
      sum += image(i)
    }
    gray.reverse.toArray
  }
}


//Tが可変の変数をうけとれるように
class Stack[T]() {
  var x = List[T]()

  def push(a: T) {
    x = a :: x
  }

  def pop() = {
    var t = x.head
    x = x.tail
    t
  }
  
  def len() = x.size
  def head() = x.head

  def p(i: Int) = x(i)

  def reset() {
    x = List[T]()
  }
}
