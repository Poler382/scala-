object t {

  def ch()={
    val x = Array(Array(1f,2f,3f),Array(4f,5f,6f))//xn = 2,yn = 3
    val af = new Affine(2,3)
    af.W = Array(1,1,1,1,1,1)
    af.forward(x)

  }
}
