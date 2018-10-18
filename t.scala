object t {
  type R = Float
  def af_ch()={
    val x = Array(
      Array[R](1,2,3),
      Array[R](4,5,6),
      Array[R](7,8,9),
      Array[R](10,11,12)
    )
    val xx =  Array[R](1,2,3)
    val af = new Affine(3,3)
    af.W = Array[R](1,3,5,7,9,11,13,15,17)
    af.b =Array[R](0,0,0)
    val ans = Array(Array[R](153,183,213),Array[R](174,210,246),Array[R](195,237,279),Array[R](216,264,312))

    val d = Array(Array[R](1,5,9),Array[R](2,6,10),Array[R](3,7,11),Array[R](4,8,12))

    af.forward(x)
    af.backward(d)
   
  }

  def af_lay(batch:Int)={
    val x = Array(Array(1f,2f),Array(3f,4f),Array(5f,6f))//xn = 2,yn = 2
    val af = new Affine(3,3)
    print("af x: ")
    Checker.test_layer(af,batch,3,3)
    print("af W: ")
    Checker.test_w_layer(af,batch,3,3)
    print("af b: ")
    Checker.test_b_layer(af,batch,3,3)
 

  }

  def conv_lay()={
    
    val conv = new Convolution3D(2,3,3,3,2)
    val conv2 = new Convolution_Matver(2,2,3,3,3,2)
    
    Checker.test_layer(conv2,10,3*3*3,2*2*2)
   

  }

def conv_ch()={
    val x = Array[Float](1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,10,3,4,5,6,7,8,9,10,11)
    val xs = Array(x,x,x)
    val conv = new Convolution3D(2,3,3,3,2)
    val conv2 = new Convolution_Matver(2,2,3,3,3,2)
    conv2.K = Array[Float](1,4,1,4,1,4,1,4,2,5,2,5,2,5,2,5,3,6,3,6,3,6,3,6)
    val kn = Array[Float](1,1,1,1,2,2,2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6)
   

    val g = Array[Float](1,1,1,1,2,2,2,2)
    val gs = Array(g,g,g)
    conv2.forward(xs)
    conv2.backward(gs)
    

}
}




