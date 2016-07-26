function f = Bealesf(in)
    
x = in(1);
y = in(2);

f = (1.5 - x + x * y)^2 + (2.25 - x + x * y^2)^2 + (2.625 - x + x * y^3)^2;
