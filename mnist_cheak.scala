import kata._
object mnist_ch{
  val mn = new mnist()
  val rand = new scala.util.Random(0)

  def select_net(num:Int)={
    val lay = num match{
      case 0 =>{
        val af1 = new Affine(28*28,100)
        val rl1 = new ReLU()
        val af2 = new Affine(100,10)
        val sf = new SoftMax()
        List(af1,rl1,af2,sf)
      }
      case 1 =>{
        val c1 = new Convolution_Matver(3,3,28,28,1,1)
        val r1 = new ReLU()
        val p1 = new Pooling(2,5,26,26)
        val c2 = new Convolution_Matver(2,2,13,13,10,10)
        val r2 = new ReLU()
        val p2 = new Pooling(2,10,12,12)
        val c3 = new Convolution_Matver(3,3,6,6,10,10)
        val r3 = new ReLU()
        val p3 = new Pooling(2,10,4,4)
        val a3 = new Affine(1*26*26,10)
        val sf = new SoftMax()
        List(c1,r1,a3)
      }
    }
    lay
  }

  def mse(a:Array[Array[T]],b:Array[Array[T]])={
    var sum = Array.ofDim[T](a.size,a(0).size)
    if(a.size == b.size){
      for(i <- 0 until a.size;j <- 0 until a(0).size){
        sum(i)(j) += (a(i)(j) - b(i)(j))*(a(i)(j) - b(i)(j))
      }
    }else{println("Do not same size")}

    sum
  }

  def sub(a:Array[T],b:Array[T])={
    var sub = Array.ofDim[T](a.size)
    for(i <- 0 until a.size){
      sub(i) = a(i) - b(i)
    }
    sub
  }

  def ones(ln:Int,mode:Int)={
    val (dtrain,dtest) = mn.load_mnist("/home/share/number")
    val dn    = 60000 // 学習データ数    ★
    val tn    = 1000 // テストデータ数  ★


    var layer = select_net(mode)
    val L = new ML()
    var num = 0

    var MS_train1 = List[T]()
    var MS_test1  = List[T]()
    var AC_train1  = List[T]()
    var AC_test   = List[T]()
    var timelist  = List[T]()

    for(i <- 0 until ln){
      var err1 = 0f
      var err2 = 0f
      var a_count = 0f
      var start_l = System.currentTimeMillis

      for((x,n) <- dtrain.take(dn) ) {
        var y1 = L.forwards(layer,x)
        L.backwards(layer,sub(y1,mn.onehot(n)))      
        
        err1 += sub(y1,x).map(a => a*a).sum
        if(mn.argmax(y1) == n){
          a_count+=1
        }
      }
      L.updates(layer)
      var time = System.currentTimeMillis - start_l
      timelist ::= time
      AC_train1 ::= a_count
      L.print_result2(i,time,List("MSE","AC"),List(err1,a_count/dn*100),1)

    }

    L.savetxt1(timelist,"timelist","/home/pll20/sbt/t1")
    L.savetxt1(MS_train1,"mslist","/home/pll20/sbt/t1")
    L.savetxt1(AC_train1,"aclist","/home/pll20/sbt/t1")

  }
 
  def Batch(a:Int,mode:Int){

    val (dtrain,dtest) = mn.load_mnist("/home/share/number")

    val ln    = a // 学習回数       ★
    val dn    = 60000 // 学習データ数    ★
    val tn    = 1000 // テストデータ数  ★

    val layer = select_net(mode)
    val L = new ML()
    var num = 0

    var MS_train1 = List[T]()
    var MS_test1  = List[T]()
    var AC_train1  = List[T]()
    var AC_test   = List[T]()
    var timelist  = List[T]()

    var ys=List[Array[Array[T]]]()

    

    val xn = rand.shuffle(dtrain.toList).take(dn)
    val xs = xn.map(_._1).toArray
    val ns = xn.map(_._2).toArray
    val nn = Array.ofDim[T](dn,10)
    for(i <- 0 until nn.size){
      nn(i) = mn.onehot(ns(i))
    }

    for(i <- 0 until ln){
      var start_l = System.currentTimeMillis
      var err1 = 0f
      var y = L.forwards(layer,xs)
      var s = mse(nn,y).toArray
      L.backwards(layer,s)
      L.updates(layer)
    
      MS_train1 ::= err1
      var a_count=0
      for(i <- 0 until y.size){
        if(mn.argmax(y(i)) == mn.argmax(nn(i))){
          a_count+=1
        }
      }
      var time = System.currentTimeMillis - start_l
      timelist  ::= time
      AC_train1 ::= a_count 
      L.print_result2(i,time,List("MSE","a_count"),List(err1,a_count/dn),1)

      L.resets(layer)
    }

    L.savetxt1(timelist,"timelist","/home/pll20/sbt/t1")
    L.savetxt1(MS_train1,"MSlist","/home/pll20/sbt/t1")
    L.savetxt1(AC_train1,"AClist","/home/pll20/sbt/t1")

  }
}
