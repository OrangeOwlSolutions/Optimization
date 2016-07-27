function out = df1dim(xt, p, grad_costfunctional)
	
df = grad_costfunctional(xt);
      
out = dot(df, p);
