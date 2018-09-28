object t {

  def ch()={
    val x = Array(Array(1f,2f),Array(3f,4f),Array(5f,6f))//xn = 2,yn = 2
    val af = new Affine(2,2)
    af.W = Array(1f,2f,3f,4f)
    af.b =Array(1f,2f)


    af.forward(x)
    val d = Array(Array(6f,5f),Array(4f,3f),Array(2f,1f))
    af.backward(d)
  }

  def conv_ch()={
    val x = Array[Float](1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,10,3,4,5,6,7,8,9,10,11)
    val xs = Array(x,x,x)
    val conv = new Convolution_Matver(2,2,3,3,3,2)
    conv.K = Array(1,4,1,4,1,4,1,4,2,5,2,5,2,5,2,5,3,6,3,6,3,6,3,6)
    val kn = Array[Float](1,1,1,1,2,2,2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6)
    conv.forward(x)

    val g = Array[Float](1,1,1,1,2,2,2,2)
    conv.backward(g)

  }
     

}



