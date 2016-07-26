function [f, g] = Booth(in)

x = in(1);
y = in(2);

f = (x + 2 * y - 7)^2 + (2 * x + y - 5)^2;

g(1) = 10 * x + 8 * y - 34;

g(2) = 8 * x + 10 * y - 38;
