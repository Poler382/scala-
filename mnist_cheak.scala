object mnist_ch{
  val mn = new mnist()

 def main(args:Array[String]){
    
    val (dtrain,dtest) = mn.load_mnist("/home/share/number")
      
    
   val mode = one
   val mode2 = two
   val noise = three
   val ln    = args(0).toInt // 学習回数       ★
   val dn    = 60000 // 学習データ数    ★
   val tn    = 10000 // テストデータ数  ★

   
   var layer =
     val L = new ML()

   //aotoEncoder learning
   var num = 0

   var MS_train1 = List[T]()
   var MS_test1  = List[T]()
   var AC_train  = List[T]()
   var AC_test   = List[T]()

   for ( i <- 0 until ln){
     num = 0
     val xn = rand.shuffle(dtrain.toList).take(dn/2)
     val xs = xn.map(_._1).toArray

     val xf = xs.map(_.map(a => a*2 - 1f))

     var err1 = 0f; var err2 = 0f
     var start_l = System.currentTimeMillis
     var ys=List[Array[T]]()
     for((x,n) <- dtrain.take(dn) ) {
       val xtemp = x

       var y = L.forwards(layer,x)

       L.backwards(layer,sub(y,x))

       ys ::= y


       L.updates(layers)
       ys = ys.reverse
       err1 += sub(y,x).map(a => a*a).sum
     } 
     
     var time = System.currentTimeMillis - start_l
   }



 }
