# def adjvst_lv(lv,location,result):
#     N = [0]*8
#     n = [0]*8
#
#     #计算总分
#     for loc,re in zip(location,result):  #计算不同功能码对应的小分 2 ,1,-1
#         if loc[0] == 1:
#             n[0] = n[0]+ re
#             N[0] = N[0] + abs(re)
#         elif loc[0] == 2:
#             n[1] = n[1] + re
#             N[1] = N[1] + abs(re)
#         elif loc[0] == 3:
#             n[2] = n[2] + re
#             N[2] = N[2] + abs(re)
#         elif loc[0] == 4:
#             n[3] = n[3] + re
#             N[3] = N[3] + abs(re)
#         elif loc[0] == 5:
#             n[4] = n[4] + re
#             N[4] = N[4] + abs(re)
#         elif loc[0] == 6:
#             n[5] = n[5] + re
#             N[5] = N[5] + abs(re)
#         elif loc[0] == 7:
#             n[6] = n[6] + re
#             N[6] = N[6] + abs(re)
#         elif loc[0] == 8:
#             n[7] = n[7] + re
#             N[7] = N[7] + abs(re)
#     i = 0
#     for l,lo in zip(lv,location):
#         de =(n[i]+1)/(N[i]+1)
#         for m in range(1,len(lo)):
#             l[lo[m]] =  1 + de
#         i = i + 1
#     return lv
def adjvst_lv(lv,location,result,fc_number):
    fc_number[location[0]-1] = fc_number[location[0]-1] + 1
    if result ==0:
        high = 1
        low = fc_number[location[0]-1]+2
        de = high/low
        for i in range(1,len(location)):
            lv[location[0]-1][location[i]] = lv[location[0]-1][location[i]] + lv[location[0]-1][location[i]]*de
    elif result == -1:
        high = -2
        low = fc_number[location[0] - 1] + 2
        de = high/low
        for i in range(1, len(location)):
            lv[location[0]-1][location[i]] = lv[location[0]-1][location[i]] + lv[location[0]-1][location[i]]*de
    elif result == 1:
        high = 2
        low = fc_number[location[0] - 1] + 2
        de = high/low
        for i in range(1, len(location)):
            lv[location[0]-1][location[i]] = lv[location[0]-1][location[i]] + lv[location[0]-1][location[i]]*de
    return lv,fc_number


