function [x2,y2,t2,pol2,cam2] =maskRegionFilter(x,y,t,pol,cam,xmin,xmax,ymin,ymax)
   cond=(x>xmin)&(x<xmax)&(y>ymin)&(y<ymax);
   x2=x(~cond);
   y2=y(~cond);
   t2=t(~cond);
   pol2=pol(~cond);
   cam2=cam(~cond);
end