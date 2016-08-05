s1 = 1*1;
p1 = 1588;
f1 = 2995;
pd1 = p1/s1;
fd1 = f1/s1;

s2 = 2*2;
p2 = 5968;
f2 = 11594;
pd2 = p2/s2;
fd2 = f2/s2;

s3 = 3*5;
p3 = 21129;
f3 = 41551;
pd3 = p3/s3;
fd3 = f3/s3;

s4 = 3*3;
p4 = 12864;
f4 = 25211;
pd4 = p4/s4;
fd4 = f4/s4;

s5 = 4*4;
p5 = 22354;
f5 = 44020;
pd5 = p5/s5;
fd5 = f5/s5;

s6 = 1*2;
p6 = 3117;
f6 = 5955;
pd6 = p6/s6;
fd6 = f6/s6;

s7 = 2*1;
p7 = 3060;
f7 = 5838;
pd7 = p7/s7;
fd7 = f7/s7;

s8 = 3*2;
p8 = 8700;
f8 = 16952;
pd8 = p8/s8;
fd8 = f8/s8;

s9 = 2*3;
p9 = 8740;
f9 = 17024;
pd9 = p9/s9;
fd9 = f9/s9;

s10 = 5*5;
p10 = 34759;
f10 = 68648;
pd10 = p10/s10;
fd10 = f10/s10;

s11 = 5*2;
p11 = 14295;
f11 = 27952;
pd11 = p11/s11;
fd11 = f11/s11;

s12 = 2*5;
p12 = 14308;
f12 = 28004;
pd12 = p12/s12;
fd12 = f12/s12;

s13 = 5*3;
p13 = 20959;
f13 = 41197;
pd13 = p13/s13;
fd13 = f13/s13;


pds = [ pd1, pd2, pd3, pd4, pd5, pd6, pd7, pd8, pd9, pd10, pd11, pd12, pd13]';
fds = [ fd1, fd2, fd3, fd4, fd5, fd6, fd7, fd8, fd9, fd10, fd11, fd12, fd13]';
mean(pds)
mean(fds)

s = [s1, s6, s7, s2, s8, s9, s4, s11, s12, s3, s13, s5,  s10]';
p = [p1, p6, p7, p2, p8, p9, p4, p11, p12, p3, p13, p5,  p10]';
f = [f1, f6, f7, f2, f8, f9, f4, f11, f12, f3, f13, f5, f10]';

figure(1);
plot(s, p);
title ('Point density vs Patch size');
xlabel ('Patch size');
ylabel ('Point density');
print -dpng point_density_.png

figure(2);
plot(s, f);
title ('Face density vs Patch size');
xlabel ('Patch size');
ylabel ('Face density');
print -dpng face_density_.png