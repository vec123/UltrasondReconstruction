

%%
T =  Interobserverrotations03error
mean1 = nanmean(T.Observer_1Gold_standard)
std1= nanstd(T.Observer_1Gold_standard)

mean2= nanmean(T.Observer_2Gold_standard)
std2 = nanstd(T.Observer_2Gold_standard)

mean3 = nanmean(T.Observer_3Gold_standard)
std3 = nanstd(T.Observer_3Gold_standard)

mean_g = mean([mean1, mean2, mean3] )
std_g = mean([std1, std2, std3] )
%%
T =  Interobservertranslationprojectionerror
s= (256/2)*0.6
mean1x = nanmean(abs(T.Observer_1Gold_standard_x(T.Observer_1Gold_standard_x<threshold)))*s
std1x= nanstd(abs(T.Observer_1Gold_standard_x(T.Observer_1Gold_standard_x<threshold)))*s

mean2x = nanmean(abs(T.Observer_2Gold_standard_x(T.Observer_2Gold_standard_x<threshold)))*s
std2x= nanstd(abs(T.Observer_2Gold_standard_x(T.Observer_2Gold_standard_x<threshold)))*s

mean3x = nanmean(abs(T.Observer_3Gold_standard_x(T.Observer_3Gold_standard_x<threshold)))*s
std3x= nanstd(abs(T.Observer_3Gold_standard_x(T.Observer_3Gold_standard_x<threshold)))*s

mean_gx = mean([mean1x, mean2x, mean3x] )
std_gx = mean([std1x, std2x, std3x] )

mean1y = nanmean(abs(T.Observer_1Gold_standard_y(T.Observer_1Gold_standard_y<threshold)))*s
std1y= nanstd(abs(T.Observer_1Gold_standard_y(T.Observer_1Gold_standard_y<threshold)))*s

mean2y = nanmean(abs(T.Observer_2Gold_standard_y(T.Observer_2Gold_standard_y<threshold)))*s
std2y= nanstd(abs(T.Observer_2Gold_standard_y(T.Observer_2Gold_standard_y<threshold)))*s

mean3y = nanmean(abs(T.Observer_3Gold_standard_y(T.Observer_3Gold_standard_y<threshold)))*s
std3y= nanstd(abs(T.Observer_3Gold_standard_y(T.Observer_3Gold_standard_y<threshold)))*s

mean_gy = mean([mean1y, mean2y, mean3y] )
std_gy = mean([std1y, std2y, std3y] )

mean1z = nanmean(abs(T.Observer_1Gold_standard_z(T.Observer_1Gold_standard_z<threshold)))*s
std1z= nanstd(abs(T.Observer_1Gold_standard_z(T.Observer_1Gold_standard_z<threshold)))*s

mean2z = nanmean(abs(T.Observer_2Gold_standard_z(T.Observer_2Gold_standard_z<threshold)))*s
std2z= nanstd(abs(T.Observer_2Gold_standard_z(T.Observer_2Gold_standard_z<threshold)))*s

mean3z = nanmean(abs(T.Observer_3Gold_standard_z(T.Observer_3Gold_standard_z<threshold)))*s
std3z= nanstd(abs(T.Observer_3Gold_standard_z(T.Observer_3Gold_standard_z<threshold)))*s

mean_gz = mean([mean1z, mean2z, mean3z] )
std_gz = mean([std1z, std2z, std3z] )

mean_g = mean([mean_gx, mean_gy, mean_gz] )
std_g = mean([std_gx, std_gy, std_gz] )